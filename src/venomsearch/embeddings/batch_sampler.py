"""Dynamic batch sampler for efficient ESM-2 inference.

Groups protein sequences by length to minimize wasted padding tokens,
using a token-budget approach instead of fixed batch sizes.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)


class TokenBudgetSampler:
    """Batch sampler that groups sequences by length to minimize padding waste.

    Instead of using a fixed batch size, this sampler allocates a token budget
    per batch. Sequences are sorted by length, then packed into batches where
    the total tokens (including padding) don't exceed the budget.

    This is critical for ESM-2 efficiency: a batch of [10aa, 10aa, 10aa] uses
    far less compute than a batch of [10aa, 10aa, 500aa] padded to 500.

    Attributes:
        sequences: Original list of sequences (not modified).
        max_tokens_per_batch: Maximum total tokens (sequences x max_length_in_batch).
        max_sequences_per_batch: Hard cap on sequences per batch.
        sorted_indices: Indices sorted by sequence length.
    """

    def __init__(
        self,
        sequences: list[str],
        max_tokens_per_batch: int = 4096,
        max_sequences_per_batch: int = 64,
    ) -> None:
        """Initialize the sampler.

        Args:
            sequences: List of protein sequences.
            max_tokens_per_batch: Token budget per batch.
            max_sequences_per_batch: Max sequences regardless of token budget.
        """
        self.sequences = sequences
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_sequences_per_batch = max_sequences_per_batch

        # Sort indices by sequence length for efficient packing
        lengths = np.array([len(s) for s in sequences])
        self.sorted_indices = np.argsort(lengths).tolist()
        self._lengths = lengths

    def __iter__(self) -> Iterator[list[int]]:
        """Yield batches of sequence indices.

        Each batch respects:
        1. Token budget: batch_size x max_seq_len_in_batch <= max_tokens_per_batch
        2. Sequence cap: len(batch) <= max_sequences_per_batch

        Yields:
            List of original indices for each batch.
        """
        current_batch: list[int] = []
        current_max_len = 0

        for idx in self.sorted_indices:
            seq_len = int(self._lengths[idx])

            # Compute what the token count would be if we add this sequence
            new_max_len = max(current_max_len, seq_len)
            new_batch_size = len(current_batch) + 1
            new_token_count = new_max_len * new_batch_size

            # Check if adding this sequence would exceed budgets
            would_exceed_tokens = new_token_count > self.max_tokens_per_batch
            would_exceed_seqs = new_batch_size > self.max_sequences_per_batch

            if current_batch and (would_exceed_tokens or would_exceed_seqs):
                yield current_batch
                current_batch = [idx]
                current_max_len = seq_len
            else:
                current_batch.append(idx)
                current_max_len = new_max_len

        # Yield remaining sequences
        if current_batch:
            yield current_batch

    def __len__(self) -> int:
        """Estimate the number of batches (computed lazily)."""
        return sum(1 for _ in self.__iter__())

    @property
    def stats(self) -> dict[str, object]:
        """Compute batching statistics for diagnostics.

        Returns:
            Dict with batch count, avg/min/max batch sizes, and efficiency.
        """
        batch_sizes: list[int] = []
        total_tokens_used = 0
        total_tokens_padded = 0

        for batch_indices in self:
            batch_size = len(batch_indices)
            batch_sizes.append(batch_size)
            lengths = [int(self._lengths[i]) for i in batch_indices]
            max_len = max(lengths)
            total_tokens_used += sum(lengths)
            total_tokens_padded += max_len * batch_size

        efficiency = total_tokens_used / total_tokens_padded if total_tokens_padded > 0 else 0.0

        return {
            "total_sequences": len(self.sequences),
            "total_batches": len(batch_sizes),
            "avg_batch_size": np.mean(batch_sizes) if batch_sizes else 0,
            "min_batch_size": min(batch_sizes) if batch_sizes else 0,
            "max_batch_size": max(batch_sizes) if batch_sizes else 0,
            "token_efficiency": f"{efficiency:.1%}",
            "total_tokens_real": total_tokens_used,
            "total_tokens_padded": total_tokens_padded,
        }


def create_length_sorted_batches(
    sequences: list[str],
    batch_size: int = 32,
) -> list[list[int]]:
    """Simple alternative: sort by length and create fixed-size batches.

    Less optimal than TokenBudgetSampler but simpler and deterministic.

    Args:
        sequences: List of protein sequences.
        batch_size: Fixed batch size.

    Returns:
        List of index batches, each containing up to batch_size indices.
    """
    lengths = np.array([len(s) for s in sequences])
    sorted_indices = np.argsort(lengths).tolist()

    batches: list[list[int]] = []
    for i in range(0, len(sorted_indices), batch_size):
        batches.append(sorted_indices[i : i + batch_size])

    return batches
