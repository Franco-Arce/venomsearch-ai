"""Pydantic models and shared data schemas for VenomSearch-AI.

Defines the canonical data structures used across the ETL pipeline,
embedding engine, and search modules.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

#: The 20 canonical amino acids.
CANONICAL_AMINO_ACIDS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")

#: Ambiguous or non-standard amino acid codes to reject.
AMBIGUOUS_AMINO_ACIDS: frozenset[str] = frozenset("BXZJUO")

#: UniProt keyword IDs for Tox-Prot toxin categories.
TOXIN_KEYWORDS: dict[str, str] = {
    "KW-0800": "Toxin",
    "KW-0528": "Neurotoxin",
    "KW-0123": "Cardiotoxin",
    "KW-0008": "Acetylcholine receptor inhibiting toxin",
    "KW-1214": "Ion channel impairing toxin",
    "KW-0782": "Postsynaptic neurotoxin",
    "KW-0629": "Presynaptic neurotoxin",
    "KW-0204": "Complement system impairing toxin",
    "KW-0472": "Membrane attack complex/perforin activity",
}

#: Sequence length bounds for valid toxin peptides.
MIN_SEQUENCE_LENGTH: int = 10
MAX_SEQUENCE_LENGTH: int = 500


# ──────────────────────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────────────────────


class ToxinFamily(StrEnum):
    """High-level toxin family classification derived from UniProt keywords."""

    NEUROTOXIN = "neurotoxin"
    CARDIOTOXIN = "cardiotoxin"
    HEMOTOXIN = "hemotoxin"
    CYTOTOXIN = "cytotoxin"
    ION_CHANNEL = "ion_channel_toxin"
    ANTIMICROBIAL = "antimicrobial"
    OTHER = "other"


# ──────────────────────────────────────────────────────────────
# Pydantic Models — UniProt API Response
# ──────────────────────────────────────────────────────────────


class UniProtEntry(BaseModel):
    """Validated representation of a single UniProt/Tox-Prot protein entry.

    Maps the relevant fields from the UniProt REST API JSON response
    to a strongly-typed Python model with validation.
    """

    accession: str = Field(..., description="Primary UniProt accession (e.g. P01437)")
    entry_name: str = Field(default="", description="UniProt entry name (e.g. 3SA1_NAJNA)")
    protein_name: str = Field(default="Unknown", description="Recommended protein name")
    organism: str = Field(default="Unknown", description="Source organism scientific name")
    organism_id: int = Field(default=0, description="NCBI Taxonomy ID")
    sequence: str = Field(..., min_length=1, description="Amino acid sequence (one-letter code)")
    sequence_length: int = Field(default=0, description="Sequence length in residues")
    keywords: list[str] = Field(default_factory=list, description="UniProt keyword names")
    keyword_ids: list[str] = Field(
        default_factory=list, description="UniProt keyword IDs (KW-xxxx)"
    )
    go_terms: list[str] = Field(default_factory=list, description="Gene Ontology term IDs")
    disulfide_bonds: int = Field(default=0, ge=0, description="Number of disulfide bonds")
    function_annotation: str | None = Field(
        default=None, description="Free-text function description"
    )
    subcellular_location: str | None = Field(
        default=None, description="Subcellular location annotation"
    )
    toxin_family: ToxinFamily = Field(
        default=ToxinFamily.OTHER, description="Classified toxin family"
    )
    is_fragment: bool = Field(default=False, description="Whether entry is a protein fragment")
    is_reviewed: bool = Field(default=True, description="Swiss-Prot (reviewed) vs TrEMBL")

    @field_validator("sequence")
    @classmethod
    def uppercase_sequence(cls, v: str) -> str:
        """Normalize sequence to uppercase."""
        return v.strip().upper()

    @field_validator("sequence_length", mode="before")
    @classmethod
    def compute_length(cls, v: int, info: object) -> int:
        """Auto-compute sequence length if not provided."""
        if v == 0 and hasattr(info, "data") and "sequence" in info.data:  # type: ignore[union-attr]
            return len(info.data["sequence"])  # type: ignore[union-attr]
        return v

    @property
    def sequence_hash(self) -> str:
        """MD5 hash of the sequence for deduplication."""
        return hashlib.md5(self.sequence.encode()).hexdigest()

    @property
    def has_ambiguous_residues(self) -> bool:
        """Check if the sequence contains non-canonical amino acids."""
        return bool(set(self.sequence) & AMBIGUOUS_AMINO_ACIDS)

    @property
    def is_valid_length(self) -> bool:
        """Check if sequence length is within acceptable toxin bounds."""
        return MIN_SEQUENCE_LENGTH <= len(self.sequence) <= MAX_SEQUENCE_LENGTH


# ──────────────────────────────────────────────────────────────
# Dataclasses — Search Results
# ──────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single result from the toxin similarity search."""

    rank: int
    accession: str
    protein_name: str
    organism: str
    toxin_family: str
    cosine_similarity: float
    sequence: str
    molecular_target: str | None = None
    disulfide_bonds: int = 0
    function_annotation: str | None = None


@dataclass
class SearchResponse:
    """Complete search response with query metadata."""

    query_sequence: str
    query_length: int
    results: list[SearchResult] = field(default_factory=list)
    total_indexed: int = 0
    search_time_ms: float = 0.0


# ──────────────────────────────────────────────────────────────
# Dataclasses — Pipeline Statistics
# ──────────────────────────────────────────────────────────────


@dataclass
class CleaningStats:
    """Statistics from the sequence cleaning step."""

    total_input: int = 0
    passed: int = 0
    rejected_ambiguous: int = 0
    rejected_fragments: int = 0
    rejected_too_short: int = 0
    rejected_too_long: int = 0
    rejected_duplicates: int = 0

    @property
    def total_rejected(self) -> int:
        return (
            self.rejected_ambiguous
            + self.rejected_fragments
            + self.rejected_too_short
            + self.rejected_too_long
            + self.rejected_duplicates
        )

    @property
    def pass_rate(self) -> float:
        if self.total_input == 0:
            return 0.0
        return self.passed / self.total_input


@dataclass
class IngestionStats:
    """Statistics from the full ETL ingestion pipeline."""

    total_fetched: int = 0
    total_after_cleaning: int = 0
    cleaning_stats: CleaningStats = field(default_factory=CleaningStats)
    families: dict[str, int] = field(default_factory=dict)
    organisms_count: int = 0
    avg_sequence_length: float = 0.0
