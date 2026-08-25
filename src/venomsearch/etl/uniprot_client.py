"""UniProt REST API client for Tox-Prot toxin data retrieval.

Handles paginated queries, streaming downloads, rate limiting,
and parsing of UniProt JSON responses into validated Pydantic models.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

import requests
from pydantic import ValidationError

from venomsearch.models import ToxinFamily, UniProtEntry

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────

UNIPROT_BASE_URL = "https://rest.uniprot.org"
SEARCH_ENDPOINT = f"{UNIPROT_BASE_URL}/uniprotkb/search"
STREAM_ENDPOINT = f"{UNIPROT_BASE_URL}/uniprotkb/stream"

#: Fields to request from the UniProt API for complete toxin metadata.
UNIPROT_FIELDS = ",".join([
    "accession",
    "id",
    "protein_name",
    "organism_name",
    "organism_id",
    "sequence",
    "length",
    "keyword",
    "go",
    "ft_disulfid",
    "cc_function",
    "cc_subcellular_location",
    "reviewed",
    "fragment",
])

#: Default toxin keyword queries (Tox-Prot program).
DEFAULT_TOXIN_QUERIES: dict[str, str] = {
    "toxin": "keyword:KW-0800",
    "neurotoxin": "keyword:KW-0528",
    "cardiotoxin": "keyword:KW-0123",
    "acetylcholine_receptor": "keyword:KW-0008",
}

#: Maximum number of retries for API requests.
MAX_RETRIES = 3

#: Base backoff delay in seconds for exponential retry.
BASE_BACKOFF_SECONDS = 2.0

#: Default page size for paginated requests.
DEFAULT_PAGE_SIZE = 500


# ──────────────────────────────────────────────────────────────
# Keyword → ToxinFamily mapping
# ──────────────────────────────────────────────────────────────

_KEYWORD_TO_FAMILY: dict[str, ToxinFamily] = {
    "Neurotoxin": ToxinFamily.NEUROTOXIN,
    "Postsynaptic neurotoxin": ToxinFamily.NEUROTOXIN,
    "Presynaptic neurotoxin": ToxinFamily.NEUROTOXIN,
    "Cardiotoxin": ToxinFamily.CARDIOTOXIN,
    "Hemostasis impairing toxin": ToxinFamily.HEMOTOXIN,
    "Hemorrhagic toxin": ToxinFamily.HEMOTOXIN,
    "Cytotoxin": ToxinFamily.CYTOTOXIN,
    "Ion channel impairing toxin": ToxinFamily.ION_CHANNEL,
    "Voltage-gated sodium channel impairing toxin": ToxinFamily.ION_CHANNEL,
    "Voltage-gated potassium channel impairing toxin": ToxinFamily.ION_CHANNEL,
    "Voltage-gated calcium channel impairing toxin": ToxinFamily.ION_CHANNEL,
    "Acetylcholine receptor inhibiting toxin": ToxinFamily.NEUROTOXIN,
    "Antimicrobial": ToxinFamily.ANTIMICROBIAL,
}


def classify_toxin_family(keywords: list[str]) -> ToxinFamily:
    """Classify a toxin into a family based on its UniProt keywords.

    Uses a priority-based lookup: more specific keywords take precedence.

    Args:
        keywords: List of UniProt keyword names.

    Returns:
        The most specific ToxinFamily match, or OTHER if no match.
    """
    for kw in keywords:
        if kw in _KEYWORD_TO_FAMILY:
            return _KEYWORD_TO_FAMILY[kw]
    return ToxinFamily.OTHER


# ──────────────────────────────────────────────────────────────
# API Client
# ──────────────────────────────────────────────────────────────


class UniProtClient:
    """Client for querying the UniProt REST API with pagination and retry logic.

    Attributes:
        session: Persistent HTTP session with default headers.
        base_url: UniProt API base URL.
    """

    def __init__(self, base_url: str = UNIPROT_BASE_URL) -> None:
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "User-Agent": "VenomSearch-AI/0.1.0 (Python; bioinformatics portfolio project)",
        })

    def fetch_toxins(
        self,
        query: str = "keyword:KW-0800",
        reviewed_only: bool = True,
        max_results: int | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> list[UniProtEntry]:
        """Fetch toxin entries from UniProt with pagination.

        Args:
            query: UniProt search query string (e.g. 'keyword:KW-0800').
            reviewed_only: If True, only fetch Swiss-Prot (reviewed) entries.
            max_results: Maximum number of entries to fetch. None = all.
            page_size: Number of results per page (max 500 for JSON).

        Returns:
            List of validated UniProtEntry objects.
        """
        if reviewed_only:
            query = f"({query}) AND reviewed:true"

        entries: list[UniProtEntry] = []
        url: str | None = SEARCH_ENDPOINT
        params: dict[str, Any] = {
            "query": query,
            "format": "json",
            "fields": UNIPROT_FIELDS,
            "size": min(page_size, max_results) if max_results else page_size,
        }

        page = 0
        while url is not None:
            page += 1
            logger.info("Fetching page %d (collected %d entries so far)...", page, len(entries))

            response = self._request_with_retry(url, params)
            data = response.json()

            results = data.get("results", [])
            if not results:
                break

            for raw_entry in results:
                parsed = self._parse_entry(raw_entry)
                if parsed is not None:
                    entries.append(parsed)

                if max_results and len(entries) >= max_results:
                    logger.info("Reached max_results=%d, stopping.", max_results)
                    return entries[:max_results]

            # Handle pagination via Link header
            url = self._get_next_url(response)
            params = {}  # params are encoded in the Link URL

        logger.info("Fetched %d total entries from UniProt.", len(entries))
        return entries

    def fetch_all_toxin_categories(
        self,
        reviewed_only: bool = True,
        max_per_category: int | None = None,
    ) -> list[UniProtEntry]:
        """Fetch entries across all Tox-Prot categories, deduplicating by accession.

        Args:
            reviewed_only: Only fetch Swiss-Prot entries.
            max_per_category: Max entries per category (for testing/dev).

        Returns:
            Deduplicated list of UniProtEntry objects.
        """
        all_entries: dict[str, UniProtEntry] = {}

        for category_name, query in DEFAULT_TOXIN_QUERIES.items():
            logger.info("Fetching category: %s (%s)", category_name, query)
            entries = self.fetch_toxins(
                query=query,
                reviewed_only=reviewed_only,
                max_results=max_per_category,
            )
            for entry in entries:
                if entry.accession not in all_entries:
                    all_entries[entry.accession] = entry

            logger.info(
                "Category '%s': %d entries (%d unique total)",
                category_name,
                len(entries),
                len(all_entries),
            )

        return list(all_entries.values())

    def stream_toxins(
        self,
        query: str = "keyword:KW-0800",
        reviewed_only: bool = True,
        output_path: Path | None = None,
    ) -> list[UniProtEntry]:
        """Stream large result sets using UniProt's stream endpoint.

        More efficient than pagination for >10K entries.

        Args:
            query: UniProt search query.
            reviewed_only: Filter to reviewed entries.
            output_path: Optional path to save raw JSONL snapshot.

        Returns:
            List of validated UniProtEntry objects.
        """
        if reviewed_only:
            query = f"({query}) AND reviewed:true"

        params = {
            "query": query,
            "format": "json",
            "fields": UNIPROT_FIELDS,
        }

        logger.info("Streaming from UniProt: %s", query)
        response = self._request_with_retry(STREAM_ENDPOINT, params, stream=True)
        data = response.json()

        entries: list[UniProtEntry] = []
        results = data.get("results", [])

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                for raw_entry in results:
                    f.write(json.dumps(raw_entry) + "\n")

        for raw_entry in results:
            parsed = self._parse_entry(raw_entry)
            if parsed is not None:
                entries.append(parsed)

        logger.info("Streamed %d entries.", len(entries))
        return entries

    # ── Private helpers ─────────────────────────────────────

    def _request_with_retry(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        stream: bool = False,
    ) -> requests.Response:
        """Execute an HTTP GET with exponential backoff retry.

        Respects the Retry-After header from UniProt (429 responses).

        Args:
            url: Request URL.
            params: Query parameters.
            stream: Whether to stream the response.

        Returns:
            The successful Response object.

        Raises:
            requests.HTTPError: After exhausting all retries.
        """
        last_exception: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                response = self.session.get(url, params=params, stream=stream, timeout=120)

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", BASE_BACKOFF_SECONDS))
                    logger.warning(
                        "Rate limited (429). Waiting %d seconds (attempt %d/%d).",
                        retry_after,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response

            except requests.RequestException as exc:
                last_exception = exc
                wait_time = BASE_BACKOFF_SECONDS * (2**attempt)
                logger.warning(
                    "Request failed: %s. Retrying in %.1fs (attempt %d/%d).",
                    exc,
                    wait_time,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait_time)

        raise requests.HTTPError(
            f"Failed after {MAX_RETRIES} retries. Last error: {last_exception}"
        )

    @staticmethod
    def _get_next_url(response: requests.Response) -> str | None:
        """Extract the next page URL from the Link header.

        UniProt uses cursor-based pagination with a Link header:
        `<https://rest.uniprot.org/...?cursor=xxx>; rel="next"`

        Args:
            response: HTTP response from UniProt.

        Returns:
            The next page URL, or None if this is the last page.
        """
        link_header = response.headers.get("Link", "")
        if 'rel="next"' in link_header:
            # Format: <URL>; rel="next"
            return link_header.split(";")[0].strip("<> ")
        return None

    @staticmethod
    def _parse_entry(raw: dict[str, Any]) -> UniProtEntry | None:
        """Parse a raw UniProt JSON entry into a validated UniProtEntry.

        Handles the complex nested structure of UniProt API responses.

        Args:
            raw: Raw JSON dict from UniProt API.

        Returns:
            Validated UniProtEntry, or None if parsing fails.
        """
        try:
            # Extract protein name from nested structure
            protein_name = "Unknown"
            pn = raw.get("proteinDescription", {})
            rec_name = pn.get("recommendedName")
            if rec_name:
                protein_name = rec_name.get("fullName", {}).get("value", "Unknown")
            elif pn.get("submissionNames"):
                protein_name = (
                    pn["submissionNames"][0].get("fullName", {}).get("value", "Unknown")
                )

            # Extract organism
            organism_data = raw.get("organism", {})
            organism = organism_data.get("scientificName", "Unknown")
            organism_id = organism_data.get("taxonId", 0)

            # Extract sequence
            seq_data = raw.get("sequence", {})
            sequence = seq_data.get("value", "")
            seq_length = seq_data.get("length", len(sequence))

            # Extract keywords
            kw_list = raw.get("keywords", [])
            keywords = [kw.get("name", "") for kw in kw_list if kw.get("name")]
            keyword_ids = [kw.get("id", "") for kw in kw_list if kw.get("id")]

            # Extract GO terms
            go_refs = [
                xr
                for xr in raw.get("uniProtKBCrossReferences", [])
                if xr.get("database") == "GO"
            ]
            go_terms = [ref.get("id", "") for ref in go_refs if ref.get("id")]

            # Count disulfide bonds from features
            features = raw.get("features", [])
            disulfide_bonds = sum(1 for f in features if f.get("type") == "Disulfide bond")

            # Extract function annotation from comments
            function_annotation = None
            subcellular_location = None
            for comment in raw.get("comments", []):
                if comment.get("commentType") == "FUNCTION":
                    texts = comment.get("texts", [])
                    if texts:
                        function_annotation = texts[0].get("value", "")
                elif comment.get("commentType") == "SUBCELLULAR LOCATION":
                    locs = comment.get("subcellularLocations", [])
                    if locs:
                        loc_val = locs[0].get("location", {}).get("value", "")
                        if loc_val:
                            subcellular_location = loc_val

            # Check if fragment
            is_fragment = bool(raw.get("proteinDescription", {}).get("flag"))
            flag_value = raw.get("proteinDescription", {}).get("flag", "")
            if isinstance(flag_value, str):
                is_fragment = "fragment" in flag_value.lower()

            # Classify toxin family
            toxin_family = classify_toxin_family(keywords)

            return UniProtEntry(
                accession=raw.get("primaryAccession", ""),
                entry_name=raw.get("uniProtkbId", ""),
                protein_name=protein_name,
                organism=organism,
                organism_id=organism_id,
                sequence=sequence,
                sequence_length=seq_length,
                keywords=keywords,
                keyword_ids=keyword_ids,
                go_terms=go_terms,
                disulfide_bonds=disulfide_bonds,
                function_annotation=function_annotation,
                subcellular_location=subcellular_location,
                toxin_family=toxin_family,
                is_fragment=is_fragment,
                is_reviewed=raw.get("entryType", "") == "UniProtKB reviewed (Swiss-Prot)",
            )

        except (ValidationError, KeyError, TypeError) as exc:
            accession = raw.get("primaryAccession", "UNKNOWN")
            logger.warning("Failed to parse entry %s: %s", accession, exc)
            return None
