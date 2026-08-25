"""ESM-2 protein language model encoder for sequence embeddings.

Uses Meta's ESM-2 (facebook/esm2_t6_8M_UR50D) to generate dense vector
representations of protein sequences via mean pooling over residue-level
hidden states.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

#: Default ESM-2 model: 8M parameters, 320-dim embeddings, CPU-friendly.
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"

#: Embedding dimensions per model.
MODEL_DIMENSIONS: dict[str, int] = {
    "facebook/esm2_t6_8M_UR50D": 320,
    "facebook/esm2_t12_35M_UR50D": 480,
    "facebook/esm2_t30_150M_UR50D": 640,
    "facebook/esm2_t33_650M_UR50D": 1280,
}


def _detect_device() -> torch.device:
    """Auto-detect the best available compute device.

    Priority: CUDA > MPS (Apple Silicon) > CPU.

    Returns:
        torch.device for inference.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


class ESMEncoder:
    """Encoder for protein sequences using Meta's ESM-2 language model.

    Generates fixed-dimensional embeddings by mean-pooling the last hidden
    layer representations, excluding special tokens (BOS/EOS).

    Attributes:
        model_name: HuggingFace model identifier.
        device: Compute device (CPU/CUDA/MPS).
        embedding_dim: Output embedding dimensionality.
        model: Loaded ESM-2 model.
        tokenizer: ESM-2 tokenizer.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
    ) -> None:
        """Initialize the ESM-2 encoder.

        Args:
            model_name: HuggingFace model name or local path.
            device: Device string ('auto', 'cpu', 'cuda', 'mps').
        """
        self.model_name = model_name
        self.device = _detect_device() if device == "auto" else torch.device(device)
        self.embedding_dim = MODEL_DIMENSIONS.get(model_name, 320)

        logger.info("Loading ESM-2 model: %s (dim=%d)...", model_name, self.embedding_dim)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully on %s.", self.device)

    def encode_single(self, sequence: str) -> np.ndarray:
        """Generate embedding for a single protein sequence.

        Args:
            sequence: Amino acid sequence (one-letter code, e.g. 'MAVPET...').

        Returns:
            1D numpy array of shape (embedding_dim,).
        """
        return self.encode_batch([sequence])[0]

    def encode_batch(
        self,
        sequences: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Generate embeddings for multiple sequences with batching.

        Sequences are processed in batches to manage memory. Each batch
        is dynamically padded to the longest sequence in that batch.

        Args:
            sequences: List of amino acid sequences.
            batch_size: Number of sequences per batch.
            show_progress: Whether to display a progress bar (via rich).

        Returns:
            2D numpy array of shape (n_sequences, embedding_dim).
        """
        all_embeddings: list[np.ndarray] = []
        total_batches = (len(sequences) + batch_size - 1) // batch_size

        # Optional rich progress bar
        if show_progress:
            try:
                from rich.progress import Progress

                progress = Progress()
                task_id = progress.add_task(
                    "[cyan]Encoding sequences...", total=total_batches
                )
                progress.start()
            except ImportError:
                progress = None  # type: ignore[assignment]
                show_progress = False

        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i : i + batch_size]
            batch_embeddings = self._encode_single_batch(batch_seqs)
            all_embeddings.append(batch_embeddings)

            batch_num = i // batch_size + 1
            if show_progress and progress is not None:  # type: ignore[possibly-undefined]
                progress.update(task_id, advance=1)  # type: ignore[possibly-undefined]
            elif batch_num % 10 == 0 or batch_num == total_batches:
                logger.info("Batch %d/%d complete.", batch_num, total_batches)

            # Free GPU memory between batches
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if show_progress and progress is not None:
            progress.stop()

        result = np.vstack(all_embeddings)
        logger.info(
            "Encoded %d sequences → shape %s",
            len(sequences),
            result.shape,
        )
        return result

    # ── Private helpers ─────────────────────────────────────

    def _encode_single_batch(self, sequences: list[str]) -> np.ndarray:
        """Encode a single batch of sequences.

        Args:
            sequences: Batch of amino acid sequences.

        Returns:
            2D numpy array of shape (batch_size, embedding_dim).
        """
        # Tokenize with dynamic padding to max length in this batch
        inputs = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            hidden_states = outputs.last_hidden_state  # (batch, seq_len, dim)

        # Mean pooling excluding special tokens (BOS at 0, EOS at -1)
        attention_mask = inputs.get("attention_mask")
        embeddings = self._mean_pool(hidden_states, attention_mask)

        return embeddings.cpu().numpy()

    @staticmethod
    def _mean_pool(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Mean pooling over residue representations, excluding special tokens.

        Masks out:
        - Position 0 (BOS token)
        - Last non-padding position (EOS token)
        - Padding positions

        Args:
            hidden_states: Shape (batch, seq_len, dim).
            attention_mask: Shape (batch, seq_len). 1 for real tokens, 0 for padding.

        Returns:
            Pooled embeddings of shape (batch, dim).
        """
        if attention_mask is None:
            # No padding: just exclude first and last tokens
            return hidden_states[:, 1:-1, :].mean(dim=1)

        # Create mask that excludes BOS (pos 0), EOS (last real token), and padding
        batch_size, _seq_len, dim = hidden_states.shape
        mask = attention_mask.clone()

        # Mask out BOS (position 0)
        mask[:, 0] = 0

        # Mask out EOS (last non-padding position for each sequence)
        seq_lengths = attention_mask.sum(dim=1)  # real token count per sequence
        for i in range(batch_size):
            eos_pos = seq_lengths[i].item() - 1  # last real token position
            if eos_pos > 0:
                mask[i, int(eos_pos)] = 0

        # Apply mask and compute mean
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, dim).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        token_counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()

        return sum_embeddings / token_counts

    @property
    def model_info(self) -> dict[str, object]:
        """Return model metadata for logging/display."""
        param_count = sum(p.numel() for p in self.model.parameters())
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "device": str(self.device),
            "parameters": f"{param_count / 1e6:.1f}M",
            "dtype": str(next(self.model.parameters()).dtype),
        }
