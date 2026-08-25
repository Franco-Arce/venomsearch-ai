"""ETL pipeline orchestrator for VenomSearch-AI.

Coordinates the full Extract-Transform-Load workflow:
UniProt API → Validation → Cleaning → Enrichment → Parquet export.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from venomsearch.etl.cleaner import SequenceCleaner
from venomsearch.etl.uniprot_client import UniProtClient
from venomsearch.models import IngestionStats, UniProtEntry

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Default paths
# ──────────────────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("data")
DEFAULT_RAW_DIR = DEFAULT_DATA_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_DATA_DIR / "processed"


class ETLPipeline:
    """Orchestrates the complete ETL pipeline for toxin data.

    Pipeline stages:
    1. **Extract**: Fetch entries from UniProt REST API (paginated or streaming)
    2. **Transform**: Validate with Pydantic, clean sequences, enrich metadata
    3. **Load**: Export to Parquet (partitioned by toxin_family) + raw JSONL snapshot

    Attributes:
        client: UniProt API client.
        cleaner: Sequence validation and cleaning utility.
        data_dir: Root data directory.
        raw_dir: Directory for raw JSONL snapshots.
        processed_dir: Directory for processed Parquet files.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        min_length: int = 10,
        max_length: int = 500,
        allow_selenocysteine: bool = True,
    ) -> None:
        self.client = UniProtClient()
        self.cleaner = SequenceCleaner(
            min_length=min_length,
            max_length=max_length,
            allow_selenocysteine=allow_selenocysteine,
        )
        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
        self.processed_dir = data_dir / "processed"

    def run(
        self,
        reviewed_only: bool = True,
        max_per_category: int | None = None,
        save_raw: bool = True,
    ) -> tuple[pl.DataFrame, IngestionStats]:
        """Execute the full ETL pipeline.

        Args:
            reviewed_only: Only fetch Swiss-Prot (reviewed) entries.
            max_per_category: Limit entries per category (for dev/testing).
            save_raw: Whether to save raw JSONL snapshot.

        Returns:
            Tuple of (processed Polars DataFrame, ingestion statistics).
        """
        logger.info("=" * 60)
        logger.info("VenomSearch-AI ETL Pipeline — Starting")
        logger.info("=" * 60)

        # ── Stage 1: Extract ────────────────────────────────
        logger.info("Stage 1/3: Extracting from UniProt/Tox-Prot...")
        raw_entries = self.client.fetch_all_toxin_categories(
            reviewed_only=reviewed_only,
            max_per_category=max_per_category,
        )
        logger.info("Extracted %d unique entries.", len(raw_entries))

        # Save raw snapshot
        if save_raw:
            self._save_raw_snapshot(raw_entries)

        # ── Stage 2: Transform ──────────────────────────────
        logger.info("Stage 2/3: Cleaning and validating sequences...")
        clean_entries = self.cleaner.clean(raw_entries)
        logger.info("After cleaning: %d entries.", len(clean_entries))

        # ── Stage 3: Load ───────────────────────────────────
        logger.info("Stage 3/3: Loading to Parquet...")
        df = self._entries_to_dataframe(clean_entries)
        self._save_parquet(df)

        # Build statistics
        stats = self._compute_stats(raw_entries, clean_entries, df)

        logger.info("=" * 60)
        logger.info("ETL Pipeline Complete!")
        logger.info("  Total fetched:   %d", stats.total_fetched)
        logger.info("  After cleaning:  %d", stats.total_after_cleaning)
        logger.info("  Unique organisms: %d", stats.organisms_count)
        logger.info("  Avg seq length:  %.1f aa", stats.avg_sequence_length)
        logger.info("  Families: %s", stats.families)
        logger.info("=" * 60)

        return df, stats

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _entries_to_dataframe(entries: list[UniProtEntry]) -> pl.DataFrame:
        """Convert a list of UniProtEntry models to a Polars DataFrame.

        Args:
            entries: Cleaned and validated entries.

        Returns:
            Polars DataFrame with typed columns.
        """
        records = []
        for entry in entries:
            records.append({
                "accession": entry.accession,
                "entry_name": entry.entry_name,
                "protein_name": entry.protein_name,
                "organism": entry.organism,
                "organism_id": entry.organism_id,
                "sequence": entry.sequence,
                "sequence_length": entry.sequence_length,
                "keywords": entry.keywords,
                "keyword_ids": entry.keyword_ids,
                "go_terms": entry.go_terms,
                "disulfide_bonds": entry.disulfide_bonds,
                "function_annotation": entry.function_annotation or "",
                "subcellular_location": entry.subcellular_location or "",
                "toxin_family": entry.toxin_family.value,
                "is_reviewed": entry.is_reviewed,
            })

        return pl.DataFrame(records)

    def _save_parquet(self, df: pl.DataFrame) -> None:
        """Save DataFrame to Parquet, partitioned by toxin_family.

        Args:
            df: Processed DataFrame.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Single file output (simpler for this dataset size)
        output_path = self.processed_dir / "toxins.parquet"
        df.write_parquet(output_path, compression="zstd")
        logger.info("Saved Parquet: %s (%d rows)", output_path, len(df))

        # Also save partitioned version for analysis
        for family in df["toxin_family"].unique().to_list():
            family_dir = self.processed_dir / f"toxin_family={family}"
            family_dir.mkdir(parents=True, exist_ok=True)
            family_df = df.filter(pl.col("toxin_family") == family)
            family_df.write_parquet(family_dir / "part-0.parquet", compression="zstd")
            logger.info("  Partition %s: %d rows", family, len(family_df))

    def _save_raw_snapshot(self, entries: list[UniProtEntry]) -> None:
        """Save raw entries as JSONL for reproducibility.

        Args:
            entries: Raw entries before cleaning.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        snapshot_path = self.raw_dir / f"uniprot_toxprot_{timestamp}.jsonl"

        with open(snapshot_path, "w", encoding="utf-8") as f:
            for entry in entries:
                f.write(entry.model_dump_json() + "\n")

        logger.info("Saved raw snapshot: %s (%d entries)", snapshot_path, len(entries))

    @staticmethod
    def _compute_stats(
        raw_entries: list[UniProtEntry],
        clean_entries: list[UniProtEntry],
        df: pl.DataFrame,
    ) -> IngestionStats:
        """Compute summary statistics for the pipeline run.

        Args:
            raw_entries: Entries before cleaning.
            clean_entries: Entries after cleaning.
            df: Final DataFrame.

        Returns:
            IngestionStats with computed metrics.
        """
        families = (
            df.group_by("toxin_family")
            .len()
            .sort("len", descending=True)
            .to_dict()
        )
        family_dict = dict(
            zip(families["toxin_family"], families["len"], strict=False)
        )

        return IngestionStats(
            total_fetched=len(raw_entries),
            total_after_cleaning=len(clean_entries),
            families=family_dict,
            organisms_count=df["organism"].n_unique(),
            avg_sequence_length=df["sequence_length"].mean() or 0.0,
        )
