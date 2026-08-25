"""LanceDB vector store interface for toxin embeddings.

Manages the creation, indexing, and querying of the toxin embedding
database using LanceDB's embedded vector search engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb

if TYPE_CHECKING:
    import numpy as np
    import polars as pl

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

DEFAULT_DB_PATH = "data/lancedb"
TABLE_NAME = "toxins"


class VenomVectorStore:
    """LanceDB-backed vector store for toxin embeddings.

    Provides a high-level interface for creating tables, inserting
    embeddings with metadata, and performing cosine similarity search.

    Attributes:
        db_path: Path to the LanceDB database directory.
        db: LanceDB connection.
        _table: Cached reference to the active table (avoids re-lookup issues).
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH) -> None:
        """Initialize connection to LanceDB.

        Args:
            db_path: Local path for the LanceDB data directory.
        """
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)
        self._table: Any | None = None
        logger.info("Connected to LanceDB at: %s", db_path)

    def _table_exists(self) -> bool:
        """Check if the toxins table exists in the database.

        Uses try/except with open_table for reliable detection
        across LanceDB versions.
        """
        if self._table is not None:
            return True
        try:
            self._table = self.db.open_table(TABLE_NAME)
            return True
        except Exception:
            return False

    def _get_table(self) -> Any:
        """Get the active table, raising if it doesn't exist."""
        if self._table is not None:
            return self._table
        if self._table_exists():
            return self._table
        raise RuntimeError(
            f"Table '{TABLE_NAME}' not found. Run 'venomsearch index' first."
        )

    def create_index(
        self,
        df: pl.DataFrame,
        embeddings: np.ndarray,
        overwrite: bool = True,
    ) -> None:
        """Create or replace the toxins table with embeddings and metadata.

        Args:
            df: Polars DataFrame with toxin metadata (must have 'accession' column).
            embeddings: Numpy array of shape (n_entries, embedding_dim).
            overwrite: If True, drop and recreate the table.
        """
        if len(df) != embeddings.shape[0]:
            raise ValueError(
                f"DataFrame rows ({len(df)}) != embeddings rows ({embeddings.shape[0]})"
            )

        logger.info(
            "Creating index: %d entries, embedding dim=%d",
            len(df),
            embeddings.shape[1],
        )

        # Build records for LanceDB
        records = self._build_records(df, embeddings)

        # Drop existing table if overwriting
        if overwrite and self._table_exists():
            self.db.drop_table(TABLE_NAME)
            self._table = None
            logger.info("Dropped existing table '%s'.", TABLE_NAME)

        table = self.db.create_table(TABLE_NAME, data=records)
        self._table = table  # Cache the reference
        logger.info("Created table '%s' with %d rows.", TABLE_NAME, len(records))

        # Create vector index for fast search (only for larger datasets)
        if len(records) >= 256:
            try:
                table.create_index(
                    metric="cosine",
                    num_partitions=min(16, len(records) // 16),
                    num_sub_vectors=min(16, embeddings.shape[1] // 8),
                )
                logger.info("Created IVF-PQ index with cosine metric.")
            except Exception as exc:
                logger.warning("Index creation failed (brute-force will be used): %s", exc)
        else:
            logger.info(
                "Dataset too small for ANN index (%d < 256). Using brute-force search.",
                len(records),
            )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_family: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict[str, Any]]:
        """Search for the most similar toxins by cosine similarity.

        Args:
            query_embedding: Query vector of shape (embedding_dim,).
            top_k: Number of results to return.
            filter_family: Optional toxin family filter (e.g. 'neurotoxin').
            min_score: Minimum cosine similarity score threshold.

        Returns:
            List of dicts with toxin metadata and similarity scores.

        Raises:
            RuntimeError: If the toxins table doesn't exist.
        """
        table = self._get_table()
        query = table.search(query_embedding.tolist()).metric("cosine").limit(top_k)

        if filter_family:
            query = query.where(f"toxin_family = '{filter_family}'")

        results = query.to_list()

        # Convert LanceDB distance to cosine similarity
        # LanceDB returns 1 - cosine_similarity as distance
        output: list[dict[str, Any]] = []
        for row in results:
            distance = row.get("_distance", 0.0)
            cosine_sim = 1.0 - distance

            if cosine_sim < min_score:
                continue

            output.append({
                "accession": row.get("accession", ""),
                "protein_name": row.get("protein_name", ""),
                "organism": row.get("organism", ""),
                "toxin_family": row.get("toxin_family", ""),
                "cosine_similarity": round(cosine_sim, 4),
                "sequence": row.get("sequence", ""),
                "sequence_length": row.get("sequence_length", 0),
                "disulfide_bonds": row.get("disulfide_bonds", 0),
                "function_annotation": row.get("function_annotation", ""),
            })

        return output

    def get_table_info(self) -> dict[str, Any]:
        """Return metadata about the indexed table.

        Returns:
            Dict with row count, schema info, and table name.
        """
        if not self._table_exists():
            return {"exists": False, "table_name": TABLE_NAME}

        table = self._get_table()
        row_count = table.count_rows()

        # Get embedding dim from first row
        embedding_dim = 0
        if row_count > 0:
            first_row = table.search().limit(1).to_list()
            if first_row and "vector" in first_row[0]:
                embedding_dim = len(first_row[0]["vector"])

        return {
            "exists": True,
            "table_name": TABLE_NAME,
            "total_rows": row_count,
            "embedding_dim": embedding_dim,
        }

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _build_records(
        df: pl.DataFrame,
        embeddings: np.ndarray,
    ) -> list[dict[str, Any]]:
        """Build LanceDB-compatible records from DataFrame and embeddings.

        Args:
            df: Metadata DataFrame.
            embeddings: Embedding array.

        Returns:
            List of dicts with 'vector' key and metadata fields.
        """
        columns = [
            "accession",
            "protein_name",
            "organism",
            "sequence",
            "sequence_length",
            "toxin_family",
            "disulfide_bonds",
            "function_annotation",
        ]
        # Use Polars native .to_dicts() — no pandas needed
        rows = df.select(columns).to_dicts()

        records: list[dict[str, Any]] = []
        for i, row in enumerate(rows):
            row["vector"] = embeddings[i].tolist()
            records.append(row)

        return records
