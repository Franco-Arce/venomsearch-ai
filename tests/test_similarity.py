"""Tests for the similarity search engine and vector store."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

from venomsearch.search.vector_store import VenomVectorStore

if TYPE_CHECKING:
    from pathlib import Path

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_test_df(n: int = 10) -> pl.DataFrame:
    """Create a test DataFrame with toxin metadata."""
    return pl.DataFrame({
        "accession": [f"P{i:05d}" for i in range(n)],
        "protein_name": [f"Test Toxin {i}" for i in range(n)],
        "organism": [f"Organism {i % 3}" for i in range(n)],
        "sequence": [f"{'ACDEFGHIKLMNPQRSTVWY'[i % 20:]}" + "A" * 20 for i in range(n)],
        "sequence_length": [30 + i for i in range(n)],
        "toxin_family": [["neurotoxin", "cardiotoxin", "hemotoxin"][i % 3] for i in range(n)],
        "disulfide_bonds": [i % 4 for i in range(n)],
        "function_annotation": [f"Function {i}" for i in range(n)],
    })


def _make_test_embeddings(n: int = 10, dim: int = 320) -> np.ndarray:
    """Create deterministic test embeddings."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim)).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# Tests: Vector Store
# ──────────────────────────────────────────────────────────────


class TestVenomVectorStore:
    """Test the LanceDB vector store interface."""

    @pytest.fixture
    def store_with_data(self, tmp_path: Path) -> VenomVectorStore:
        """Create a temporary vector store with test data."""
        store = VenomVectorStore(db_path=str(tmp_path / "test_lancedb"))
        df = _make_test_df(20)
        embeddings = _make_test_embeddings(20)
        store.create_index(df, embeddings)
        return store

    def test_create_index(self, tmp_path: Path) -> None:
        store = VenomVectorStore(db_path=str(tmp_path / "test_db"))
        df = _make_test_df(10)
        embeddings = _make_test_embeddings(10)

        store.create_index(df, embeddings)

        info = store.get_table_info()
        assert info["exists"] is True
        assert info["total_rows"] == 10
        assert info["embedding_dim"] == 320

    def test_mismatched_rows_raises(self, tmp_path: Path) -> None:
        """DataFrame and embeddings must have same number of rows."""
        store = VenomVectorStore(db_path=str(tmp_path / "test_db"))
        df = _make_test_df(10)
        embeddings = _make_test_embeddings(5)  # Mismatch!

        with pytest.raises(ValueError, match="DataFrame rows"):
            store.create_index(df, embeddings)

    def test_search_returns_results(self, store_with_data: VenomVectorStore) -> None:
        # Use a random query vector
        query = _make_test_embeddings(1, 320)[0]
        results = store_with_data.search(query, top_k=5)

        assert len(results) <= 5
        assert all("accession" in r for r in results)
        assert all("cosine_similarity" in r for r in results)

    def test_search_top_k(self, store_with_data: VenomVectorStore) -> None:
        """Should return exactly top_k results."""
        query = _make_test_embeddings(1, 320)[0]
        results = store_with_data.search(query, top_k=3)
        assert len(results) == 3

    def test_self_similarity_is_highest(self, tmp_path: Path) -> None:
        """Searching with an indexed vector should return itself with score ~1.0."""
        store = VenomVectorStore(db_path=str(tmp_path / "test_db"))
        df = _make_test_df(5)
        embeddings = _make_test_embeddings(5)

        # Normalize embeddings for clean cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normed = embeddings / norms

        store.create_index(df, embeddings_normed)

        # Search with the first vector
        results = store.search(embeddings_normed[0], top_k=1)
        assert len(results) == 1
        assert results[0]["accession"] == "P00000"
        assert results[0]["cosine_similarity"] > 0.99

    def test_search_results_ordered(self, store_with_data: VenomVectorStore) -> None:
        """Results should be ordered by similarity (descending)."""
        query = _make_test_embeddings(1, 320)[0]
        results = store_with_data.search(query, top_k=5)

        scores = [r["cosine_similarity"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_filter_by_family(self, store_with_data: VenomVectorStore) -> None:
        """Family filter should only return matching entries."""
        query = _make_test_embeddings(1, 320)[0]
        results = store_with_data.search(query, top_k=10, filter_family="neurotoxin")

        assert all(r["toxin_family"] == "neurotoxin" for r in results)

    def test_overwrite_replaces_table(self, tmp_path: Path) -> None:
        store = VenomVectorStore(db_path=str(tmp_path / "test_db"))

        # Create with 10 entries
        store.create_index(_make_test_df(10), _make_test_embeddings(10))
        assert store.get_table_info()["total_rows"] == 10

        # Overwrite with 5 entries
        store.create_index(_make_test_df(5), _make_test_embeddings(5), overwrite=True)
        assert store.get_table_info()["total_rows"] == 5

    def test_table_info_nonexistent(self, tmp_path: Path) -> None:
        store = VenomVectorStore(db_path=str(tmp_path / "empty_db"))
        info = store.get_table_info()
        assert info["exists"] is False

    def test_search_nonexistent_table_raises(self, tmp_path: Path) -> None:
        store = VenomVectorStore(db_path=str(tmp_path / "empty_db"))
        query = _make_test_embeddings(1, 320)[0]

        with pytest.raises(RuntimeError, match="not found"):
            store.search(query)
