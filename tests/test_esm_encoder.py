"""Tests for the ESM-2 encoder module.

These tests verify the encoder interface and logic without requiring
a GPU or downloading the full model. The actual model loading tests
are marked as 'slow' and can be skipped in CI with `-m "not slow"`.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from venomsearch.embeddings.esm_encoder import ESMEncoder, _detect_device

# ──────────────────────────────────────────────────────────────
# Tests: Device detection
# ──────────────────────────────────────────────────────────────


class TestDeviceDetection:
    """Test automatic device selection."""

    @patch("torch.cuda.is_available", return_value=False)
    def test_falls_back_to_cpu(self, mock_cuda: MagicMock) -> None:
        device = _detect_device()
        assert device.type == "cpu"

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="NVIDIA RTX 3060")
    def test_selects_cuda(self, mock_name: MagicMock, mock_cuda: MagicMock) -> None:
        device = _detect_device()
        assert device.type == "cuda"


# ──────────────────────────────────────────────────────────────
# Tests: Mean pooling
# ──────────────────────────────────────────────────────────────


class TestMeanPooling:
    """Test the mean pooling logic independently of the model."""

    def test_without_mask(self) -> None:
        """Mean pool over all non-special tokens (exclude first and last)."""
        # Shape: (batch=1, seq_len=5, dim=4)
        hidden = torch.tensor([
            [
                [1.0, 0.0, 0.0, 0.0],  # BOS — should be excluded
                [0.0, 1.0, 0.0, 0.0],  # token 1
                [0.0, 0.0, 1.0, 0.0],  # token 2
                [0.0, 0.0, 0.0, 1.0],  # token 3
                [2.0, 2.0, 2.0, 2.0],  # EOS — should be excluded
            ]
        ])

        result = ESMEncoder._mean_pool(hidden, attention_mask=None)

        # Mean of tokens 1-3: (0+0+0)/3, (1+0+0)/3, (0+1+0)/3, (0+0+1)/3
        expected = torch.tensor([[0.0, 1 / 3, 1 / 3, 1 / 3]])
        assert result.shape == (1, 4)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_with_padding_mask(self) -> None:
        """Mean pool should ignore padding tokens."""
        # Shape: (batch=2, seq_len=4, dim=2)
        hidden = torch.tensor([
            [
                [1.0, 1.0],  # BOS
                [2.0, 4.0],  # token
                [6.0, 8.0],  # EOS
                [0.0, 0.0],  # PAD
            ],
            [
                [1.0, 1.0],  # BOS
                [3.0, 5.0],  # EOS
                [0.0, 0.0],  # PAD
                [0.0, 0.0],  # PAD
            ],
        ])
        mask = torch.tensor([
            [1, 1, 1, 0],  # 3 real tokens
            [1, 1, 0, 0],  # 2 real tokens
        ])

        result = ESMEncoder._mean_pool(hidden, attention_mask=mask)
        assert result.shape == (2, 2)

        # Seq 1: BOS(0) masked, token=idx1 kept, EOS(idx2) masked → only idx1
        # So result[0] = [2.0, 4.0]
        assert torch.allclose(result[0], torch.tensor([2.0, 4.0]), atol=1e-5)

    def test_deterministic(self) -> None:
        """Same input should always produce same output."""
        hidden = torch.randn(2, 10, 8)
        mask = torch.ones(2, 10)

        r1 = ESMEncoder._mean_pool(hidden, mask)
        r2 = ESMEncoder._mean_pool(hidden, mask)
        assert torch.allclose(r1, r2)


# ──────────────────────────────────────────────────────────────
# Tests: Encoder integration (requires model download)
# ──────────────────────────────────────────────────────────────


@pytest.mark.slow
class TestEncoderIntegration:
    """Integration tests that load the actual ESM-2 model.

    Skip with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    def encoder(self) -> ESMEncoder:
        return ESMEncoder(model_name="facebook/esm2_t6_8M_UR50D", device="cpu")

    def test_single_encoding_shape(self, encoder: ESMEncoder) -> None:
        result = encoder.encode_single("ACDEFGHIKLMNPQRSTVWY")
        assert isinstance(result, np.ndarray)
        assert result.shape == (320,)

    def test_batch_encoding_shape(self, encoder: ESMEncoder) -> None:
        seqs = ["ACDEFGHIKLMNPQRSTVWY", "MKTLLLTLVVVTIVCLDLGYT"]
        result = encoder.encode_batch(seqs, batch_size=2, show_progress=False)
        assert result.shape == (2, 320)

    def test_short_sequence(self, encoder: ESMEncoder) -> None:
        """Very short sequences should still produce valid embeddings."""
        result = encoder.encode_single("ACF")
        assert result.shape == (320,)
        assert not np.any(np.isnan(result))

    def test_determinism(self, encoder: ESMEncoder) -> None:
        """Same sequence should always produce the same embedding."""
        seq = "ACDEFGHIKLMNPQRSTVWY"
        r1 = encoder.encode_single(seq)
        r2 = encoder.encode_single(seq)
        np.testing.assert_array_almost_equal(r1, r2, decimal=5)

    def test_different_sequences_different_embeddings(self, encoder: ESMEncoder) -> None:
        """Different sequences should produce different embeddings."""
        r1 = encoder.encode_single("ACDEFGHIKLMNPQRSTVWY")
        r2 = encoder.encode_single("WYACDEFGHIKLMNPQRSTV")
        assert not np.allclose(r1, r2)

    def test_model_info(self, encoder: ESMEncoder) -> None:
        info = encoder.model_info
        assert info["embedding_dim"] == 320
        assert "8" in str(info["parameters"])  # "8.0M"
