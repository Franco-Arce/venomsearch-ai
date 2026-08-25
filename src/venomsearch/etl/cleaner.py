"""Sequence validation and cleaning for protein toxin data.

Implements FASTA sequence quality control: alphabet validation,
fragment filtering, length filtering, deduplication, and statistics.
"""

from __future__ import annotations

import logging

from venomsearch.models import (
    AMBIGUOUS_AMINO_ACIDS,
    CANONICAL_AMINO_ACIDS,
    MAX_SEQUENCE_LENGTH,
    MIN_SEQUENCE_LENGTH,
    CleaningStats,
    UniProtEntry,
)

logger = logging.getLogger(__name__)


class SequenceCleaner:
    """Validates and filters protein sequences for downstream processing.

    Applies a configurable series of quality checks to remove sequences
    that would cause issues with ESM-2 embedding or biological analysis.

    Attributes:
        min_length: Minimum sequence length to accept.
        max_length: Maximum sequence length to accept.
        allow_selenocysteine: If True, convert U→C instead of rejecting.
        stats: Accumulated cleaning statistics.
    """

    def __init__(
        self,
        min_length: int = MIN_SEQUENCE_LENGTH,
        max_length: int = MAX_SEQUENCE_LENGTH,
        allow_selenocysteine: bool = True,
    ) -> None:
        self.min_length = min_length
        self.max_length = max_length
        self.allow_selenocysteine = allow_selenocysteine
        self.stats = CleaningStats()

    def clean(self, entries: list[UniProtEntry]) -> list[UniProtEntry]:
        """Apply all cleaning steps to a list of UniProt entries.

        Cleaning pipeline:
        1. Remove fragments
        2. Normalize sequences (uppercase, strip whitespace)
        3. Handle selenocysteine (U→C) if configured
        4. Reject sequences with ambiguous residues (B, Z, X, J, O)
        5. Filter by length
        6. Deduplicate by sequence hash

        Args:
            entries: Raw list of UniProtEntry objects from the API.

        Returns:
            Cleaned list with only valid, unique entries.
        """
        self.stats = CleaningStats(total_input=len(entries))
        logger.info("Starting cleaning of %d entries...", len(entries))

        # Step 1: Remove fragments
        non_fragments = self._filter_fragments(entries)

        # Step 2-4: Validate sequences
        valid_seqs = self._validate_sequences(non_fragments)

        # Step 5: Filter by length
        valid_length = self._filter_by_length(valid_seqs)

        # Step 6: Deduplicate
        unique = self._deduplicate(valid_length)

        self.stats.passed = len(unique)
        logger.info(
            "Cleaning complete: %d → %d entries (%.1f%% pass rate)",
            self.stats.total_input,
            self.stats.passed,
            self.stats.pass_rate * 100,
        )
        self._log_stats()
        return unique

    def validate_single(self, sequence: str) -> tuple[bool, str]:
        """Validate a single sequence string.

        Args:
            sequence: Amino acid sequence (one-letter code).

        Returns:
            Tuple of (is_valid, reason). reason is empty if valid.
        """
        seq = sequence.strip().upper()

        if not seq:
            return False, "Empty sequence"

        if len(seq) < self.min_length:
            return False, f"Too short ({len(seq)} < {self.min_length})"

        if len(seq) > self.max_length:
            return False, f"Too long ({len(seq)} > {self.max_length})"

        # Handle selenocysteine
        if self.allow_selenocysteine:
            seq = seq.replace("U", "C")

        # Check for ambiguous residues
        non_canonical = set(seq) - CANONICAL_AMINO_ACIDS
        if non_canonical:
            ambiguous = non_canonical & AMBIGUOUS_AMINO_ACIDS
            unknown = non_canonical - AMBIGUOUS_AMINO_ACIDS
            if ambiguous or unknown:
                return False, f"Contains non-canonical residues: {non_canonical}"

        return True, ""

    # ── Private filtering steps ─────────────────────────────

    def _filter_fragments(self, entries: list[UniProtEntry]) -> list[UniProtEntry]:
        """Remove entries flagged as protein fragments."""
        result = []
        for entry in entries:
            if entry.is_fragment:
                self.stats.rejected_fragments += 1
            else:
                result.append(entry)
        logger.debug("Fragment filter: %d → %d", len(entries), len(result))
        return result

    def _validate_sequences(self, entries: list[UniProtEntry]) -> list[UniProtEntry]:
        """Validate amino acid sequences: normalize and check alphabet."""
        result = []
        for entry in entries:
            seq = entry.sequence.strip().upper()

            # Handle selenocysteine: U → C (conservative substitution)
            if self.allow_selenocysteine:
                seq = seq.replace("U", "C")

            # Check for remaining ambiguous/non-standard residues
            non_canonical = set(seq) - CANONICAL_AMINO_ACIDS
            if non_canonical:
                self.stats.rejected_ambiguous += 1
                logger.debug(
                    "Rejected %s: non-canonical residues %s",
                    entry.accession,
                    non_canonical,
                )
                continue

            # Update the sequence with the cleaned version
            entry.sequence = seq
            entry.sequence_length = len(seq)
            result.append(entry)

        logger.debug("Sequence validation: %d → %d", len(entries), len(result))
        return result

    def _filter_by_length(self, entries: list[UniProtEntry]) -> list[UniProtEntry]:
        """Filter sequences outside the acceptable length range."""
        result = []
        for entry in entries:
            seq_len = len(entry.sequence)
            if seq_len < self.min_length:
                self.stats.rejected_too_short += 1
            elif seq_len > self.max_length:
                self.stats.rejected_too_long += 1
            else:
                result.append(entry)
        logger.debug("Length filter [%d, %d]: %d → %d",
                      self.min_length, self.max_length, len(entries), len(result))
        return result

    def _deduplicate(self, entries: list[UniProtEntry]) -> list[UniProtEntry]:
        """Remove duplicate sequences, keeping the first occurrence."""
        seen_hashes: set[str] = set()
        unique: list[UniProtEntry] = []

        for entry in entries:
            seq_hash = entry.sequence_hash
            if seq_hash in seen_hashes:
                self.stats.rejected_duplicates += 1
                logger.debug("Duplicate sequence: %s", entry.accession)
            else:
                seen_hashes.add(seq_hash)
                unique.append(entry)

        logger.debug("Deduplication: %d → %d", len(entries), len(unique))
        return unique

    def _log_stats(self) -> None:
        """Log detailed cleaning statistics."""
        s = self.stats
        logger.info("─── Cleaning Statistics ───")
        logger.info("  Total input:         %d", s.total_input)
        logger.info("  Passed:              %d (%.1f%%)", s.passed, s.pass_rate * 100)
        logger.info("  Rejected fragments:  %d", s.rejected_fragments)
        logger.info("  Rejected ambiguous:  %d", s.rejected_ambiguous)
        logger.info("  Rejected too short:  %d", s.rejected_too_short)
        logger.info("  Rejected too long:   %d", s.rejected_too_long)
        logger.info("  Rejected duplicates: %d", s.rejected_duplicates)
        logger.info("───────────────────────────")
