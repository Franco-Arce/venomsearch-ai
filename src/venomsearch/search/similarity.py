"""Similarity search engine for toxin sequences.

Combines the ESM-2 encoder with the LanceDB vector store to provide
end-to-end sequence-to-results search functionality.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from venomsearch.embeddings.esm_encoder import ESMEncoder
from venomsearch.models import SearchResponse, SearchResult
from venomsearch.search.vector_store import VenomVectorStore

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


class ToxinSearchEngine:
    """End-to-end search engine for toxin similarity.

    Takes a raw amino acid sequence, computes its ESM-2 embedding on the fly,
    and queries the LanceDB vector store for the most similar known toxins.

    Attributes:
        encoder: ESM-2 encoder for generating query embeddings.
        store: LanceDB vector store with indexed toxin embeddings.
    """

    def __init__(
        self,
        encoder: ESMEncoder,
        store: VenomVectorStore,
    ) -> None:
        """Initialize the search engine.

        Args:
            encoder: Pre-loaded ESM-2 encoder.
            store: Pre-connected LanceDB vector store.
        """
        self.encoder = encoder
        self.store = store

    @classmethod
    def from_paths(
        cls,
        model_name: str = "facebook/esm2_t6_8M_UR50D",
        db_path: str = "data/lancedb",
        device: str = "auto",
    ) -> ToxinSearchEngine:
        """Factory method to create a search engine from configuration paths.

        Args:
            model_name: ESM-2 model identifier.
            db_path: Path to LanceDB database.
            device: Compute device ('auto', 'cpu', 'cuda', 'mps').

        Returns:
            Initialized ToxinSearchEngine.
        """
        encoder = ESMEncoder(model_name=model_name, device=device)
        store = VenomVectorStore(db_path=db_path)
        return cls(encoder=encoder, store=store)

    def search(
        self,
        query_sequence: str,
        top_k: int = 5,
        filter_family: str | None = None,
        min_score: float = 0.0,
    ) -> SearchResponse:
        """Search for toxins similar to a query sequence.

        Pipeline:
        1. Compute ESM-2 embedding for the query sequence
        2. Query LanceDB for top-k nearest neighbors (cosine)
        3. Build and return ranked SearchResponse

        Args:
            query_sequence: Amino acid sequence (one-letter code).
            top_k: Number of results to return.
            filter_family: Optional toxin family filter.
            min_score: Minimum cosine similarity threshold.

        Returns:
            SearchResponse with ranked results and metadata.
        """
        start_time = time.perf_counter()

        # Step 1: Encode query
        logger.info("Encoding query sequence (%d aa)...", len(query_sequence))
        query_embedding = self.encoder.encode_single(query_sequence)

        # Step 2: Search vector store
        raw_results = self.store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_family=filter_family,
            min_score=min_score,
        )

        # Step 3: Build response
        results = [
            SearchResult(
                rank=i + 1,
                accession=r["accession"],
                protein_name=r["protein_name"],
                organism=r["organism"],
                toxin_family=r["toxin_family"],
                cosine_similarity=r["cosine_similarity"],
                sequence=r["sequence"],
                disulfide_bonds=r.get("disulfide_bonds", 0),
                function_annotation=r.get("function_annotation"),
            )
            for i, r in enumerate(raw_results)
        ]

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        table_info = self.store.get_table_info()

        response = SearchResponse(
            query_sequence=query_sequence,
            query_length=len(query_sequence),
            results=results,
            total_indexed=table_info.get("total_rows", 0),  # type: ignore[arg-type]
            search_time_ms=round(elapsed_ms, 2),
        )

        logger.info(
            "Search complete: %d results in %.1fms (indexed: %d)",
            len(results),
            elapsed_ms,
            response.total_indexed,
        )
        return response

    def search_fasta_file(
        self,
        fasta_path: str | Path,
        top_k: int = 5,
        filter_family: str | None = None,
    ) -> list[SearchResponse]:
        """Search for similar toxins for each sequence in a FASTA file.

        Args:
            fasta_path: Path to a FASTA file with one or more sequences.
            top_k: Number of results per query.
            filter_family: Optional toxin family filter.

        Returns:
            List of SearchResponse objects, one per query sequence.
        """
        sequences = self._parse_fasta(fasta_path)
        logger.info("Loaded %d sequences from %s", len(sequences), fasta_path)

        responses: list[SearchResponse] = []
        for seq_id, sequence in sequences:
            logger.info("Searching for: %s (%d aa)", seq_id, len(sequence))
            response = self.search(
                query_sequence=sequence,
                top_k=top_k,
                filter_family=filter_family,
            )
            responses.append(response)

        return responses

    @staticmethod
    def _parse_fasta(fasta_path: str | Path) -> list[tuple[str, str]]:
        """Parse a simple FASTA file into (id, sequence) tuples.

        Args:
            fasta_path: Path to the FASTA file.

        Returns:
            List of (header_id, sequence) tuples.
        """
        sequences: list[tuple[str, str]] = []
        current_id = ""
        current_seq: list[str] = []

        with open(fasta_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_id and current_seq:
                        sequences.append((current_id, "".join(current_seq)))
                    current_id = line[1:].split()[0]  # Take first word after >
                    current_seq = []
                elif line:
                    current_seq.append(line.upper())

        if current_id and current_seq:
            sequences.append((current_id, "".join(current_seq)))

        return sequences
