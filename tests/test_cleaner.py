"""Tests for the sequence cleaning and validation module."""

from __future__ import annotations

import pytest

from venomsearch.etl.cleaner import SequenceCleaner
from venomsearch.models import ToxinFamily, UniProtEntry

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_entry(
    sequence: str,
    accession: str = "P00001",
    is_fragment: bool = False,
) -> UniProtEntry:
    """Create a minimal UniProtEntry for testing."""
    return UniProtEntry(
        accession=accession,
        sequence=sequence,
        sequence_length=len(sequence),
        is_fragment=is_fragment,
        toxin_family=ToxinFamily.NEUROTOXIN,
    )


# ──────────────────────────────────────────────────────────────
# Tests: Ambiguous residues
# ──────────────────────────────────────────────────────────────


class TestAmbiguousResidues:
    """Sequences with non-canonical amino acids should be rejected."""

    @pytest.mark.parametrize(
        "seq,description",
        [
            ("ACDEFGHIKLXNPQRSTVWY", "contains X (unknown)"),
            ("ACDEFGHIKLBNPQRSTVWY", "contains B (Asx)"),
            ("ACDEFGHIKLZNPQRSTVWY", "contains Z (Glx)"),
            ("ACDEFGHIKLJNPQRSTVWY", "contains J (Xle)"),
            ("ACDEFGHIKLONPQRSTVWY", "contains O (pyrrolysine)"),
        ],
    )
    def test_rejects_ambiguous(self, seq: str, description: str) -> None:
        cleaner = SequenceCleaner()
        entries = [_make_entry(seq)]
        result = cleaner.clean(entries)
        assert len(result) == 0, f"Should reject sequence that {description}"

    def test_selenocysteine_converted_when_allowed(self) -> None:
        """U (selenocysteine) should be converted to C when allow_selenocysteine=True."""
        cleaner = SequenceCleaner(allow_selenocysteine=True)
        seq = "ACDEFGHIKLUNPQRSTVWY"
        entries = [_make_entry(seq)]
        result = cleaner.clean(entries)
        assert len(result) == 1
        assert "U" not in result[0].sequence
        assert "C" in result[0].sequence

    def test_selenocysteine_rejected_when_disallowed(self) -> None:
        """U should be rejected when allow_selenocysteine=False."""
        cleaner = SequenceCleaner(allow_selenocysteine=False)
        seq = "ACDEFGHIKLUNPQRSTVWY"
        entries = [_make_entry(seq)]
        result = cleaner.clean(entries)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────
# Tests: Fragment filtering
# ──────────────────────────────────────────────────────────────


class TestFragmentFiltering:
    """Protein fragments should be excluded."""

    def test_fragments_removed(self) -> None:
        entries = [
            _make_entry("ACDEFGHIKLMNPQRSTVWY", accession="P001", is_fragment=True),
            _make_entry("ACDEFGHIKLMNPQRSTVWY", accession="P002", is_fragment=False),
        ]
        cleaner = SequenceCleaner()
        result = cleaner.clean(entries)
        assert len(result) == 1
        assert result[0].accession == "P002"

    def test_stats_count_fragments(self) -> None:
        entries = [
            _make_entry("ACDEFGHIKLMNPQRSTVWY", is_fragment=True),
            _make_entry("ACDEFGHIKLMNPQRSTVWY", is_fragment=True),
            _make_entry("ACDEFGHIKLMNPQRSTVWY", is_fragment=False),
        ]
        cleaner = SequenceCleaner()
        cleaner.clean(entries)
        assert cleaner.stats.rejected_fragments == 2


# ──────────────────────────────────────────────────────────────
# Tests: Length filtering
# ──────────────────────────────────────────────────────────────


class TestLengthFiltering:
    """Sequences too short or too long should be excluded."""

    def test_too_short(self) -> None:
        cleaner = SequenceCleaner(min_length=10)
        entries = [_make_entry("ACDEF")]  # 5 aa < 10
        result = cleaner.clean(entries)
        assert len(result) == 0
        assert cleaner.stats.rejected_too_short == 1

    def test_too_long(self) -> None:
        cleaner = SequenceCleaner(max_length=50)
        long_seq = "A" * 100
        entries = [_make_entry(long_seq)]
        result = cleaner.clean(entries)
        assert len(result) == 0
        assert cleaner.stats.rejected_too_long == 1

    def test_boundary_accepted(self) -> None:
        """Sequences exactly at min/max boundary should be accepted."""
        cleaner = SequenceCleaner(min_length=10, max_length=20)
        entries = [
            _make_entry("ACDEFGHIKL"),              # exactly 10
            _make_entry("ACDEFGHIKLMNPQRSTVWY"),    # exactly 20
        ]
        result = cleaner.clean(entries)
        assert len(result) == 2

    def test_empty_sequence_rejected(self) -> None:
        """Empty sequences should fail Pydantic validation or length filter."""
        cleaner = SequenceCleaner()
        # Pydantic requires min_length=1, so we test with very short
        entries = [_make_entry("A")]  # 1 aa < default min 10
        result = cleaner.clean(entries)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────
# Tests: Deduplication
# ──────────────────────────────────────────────────────────────


class TestDeduplication:
    """Duplicate sequences (by content, not accession) should be removed."""

    def test_exact_duplicates(self) -> None:
        seq = "ACDEFGHIKLMNPQRSTVWY"
        entries = [
            _make_entry(seq, accession="P001"),
            _make_entry(seq, accession="P002"),
            _make_entry(seq, accession="P003"),
        ]
        cleaner = SequenceCleaner()
        result = cleaner.clean(entries)
        assert len(result) == 1
        assert result[0].accession == "P001"  # First occurrence kept
        assert cleaner.stats.rejected_duplicates == 2

    def test_different_sequences_kept(self) -> None:
        entries = [
            _make_entry("ACDEFGHIKLMNPQRSTVWY", accession="P001"),
            _make_entry("WYACDEFGHIKLMNPQRSTV", accession="P002"),
        ]
        cleaner = SequenceCleaner()
        result = cleaner.clean(entries)
        assert len(result) == 2


# ──────────────────────────────────────────────────────────────
# Tests: validate_single
# ──────────────────────────────────────────────────────────────


class TestValidateSingle:
    """Test the single-sequence validation method."""

    def test_valid_sequence(self) -> None:
        cleaner = SequenceCleaner()
        is_valid, reason = cleaner.validate_single("ACDEFGHIKLMNPQRSTVWY")
        assert is_valid
        assert reason == ""

    def test_empty_string(self) -> None:
        cleaner = SequenceCleaner()
        is_valid, reason = cleaner.validate_single("")
        assert not is_valid
        assert "Empty" in reason

    def test_too_short_single(self) -> None:
        cleaner = SequenceCleaner(min_length=10)
        is_valid, reason = cleaner.validate_single("ACF")
        assert not is_valid
        assert "short" in reason.lower()

    def test_ambiguous_single(self) -> None:
        cleaner = SequenceCleaner()
        is_valid, reason = cleaner.validate_single("ACDEFXGHIKLM")
        assert not is_valid
        assert "non-canonical" in reason.lower()

    def test_lowercase_normalized(self) -> None:
        cleaner = SequenceCleaner()
        is_valid, _ = cleaner.validate_single("acdefghiklmnpqrstvwy")
        assert is_valid


# ──────────────────────────────────────────────────────────────
# Tests: Statistics
# ──────────────────────────────────────────────────────────────


class TestCleaningStats:
    """Verify cleaning statistics are tracked correctly."""

    def test_pass_rate(self) -> None:
        entries = [
            _make_entry("ACDEFGHIKLMNPQRSTVWY", accession="P001"),
            _make_entry("XXXXXXXXXXXXXNPQRST", accession="P002"),  # ambiguous
            _make_entry("ACF", accession="P003"),  # too short
        ]
        cleaner = SequenceCleaner()
        cleaner.clean(entries)
        assert cleaner.stats.total_input == 3
        assert cleaner.stats.passed == 1
        assert cleaner.stats.pass_rate == pytest.approx(1 / 3, abs=0.01)

    def test_total_rejected(self) -> None:
        entries = [
            _make_entry("ACDEFGHIKLMNPQRSTVWY", is_fragment=True),  # fragment
            _make_entry("ACDEFXHIKLMNPQRSTVWY"),  # ambiguous
        ]
        cleaner = SequenceCleaner()
        cleaner.clean(entries)
        assert cleaner.stats.total_rejected == 2

    def test_zero_input(self) -> None:
        cleaner = SequenceCleaner()
        cleaner.clean([])
        assert cleaner.stats.total_input == 0
        assert cleaner.stats.passed == 0
        assert cleaner.stats.pass_rate == 0.0
