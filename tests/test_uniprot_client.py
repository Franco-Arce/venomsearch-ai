"""Tests for the UniProt API client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import requests

from venomsearch.etl.uniprot_client import (
    UniProtClient,
    classify_toxin_family,
)
from venomsearch.models import ToxinFamily

# ──────────────────────────────────────────────────────────────
# Tests: Toxin family classification
# ──────────────────────────────────────────────────────────────


class TestClassifyToxinFamily:
    """Test keyword-to-family classification logic."""

    def test_neurotoxin(self) -> None:
        assert classify_toxin_family(["Neurotoxin"]) == ToxinFamily.NEUROTOXIN

    def test_postsynaptic_neurotoxin(self) -> None:
        assert classify_toxin_family(["Postsynaptic neurotoxin"]) == ToxinFamily.NEUROTOXIN

    def test_cardiotoxin(self) -> None:
        assert classify_toxin_family(["Cardiotoxin"]) == ToxinFamily.CARDIOTOXIN

    def test_hemotoxin(self) -> None:
        assert classify_toxin_family(["Hemostasis impairing toxin"]) == ToxinFamily.HEMOTOXIN

    def test_ion_channel(self) -> None:
        result = classify_toxin_family(["Voltage-gated sodium channel impairing toxin"])
        assert result == ToxinFamily.ION_CHANNEL

    def test_acetylcholine_is_neurotoxin(self) -> None:
        result = classify_toxin_family(["Acetylcholine receptor inhibiting toxin"])
        assert result == ToxinFamily.NEUROTOXIN

    def test_unknown_defaults_to_other(self) -> None:
        assert classify_toxin_family(["Unknown keyword"]) == ToxinFamily.OTHER

    def test_empty_keywords(self) -> None:
        assert classify_toxin_family([]) == ToxinFamily.OTHER

    def test_priority_first_match(self) -> None:
        """First matching keyword wins."""
        result = classify_toxin_family(["Neurotoxin", "Cardiotoxin"])
        assert result == ToxinFamily.NEUROTOXIN


# ──────────────────────────────────────────────────────────────
# Tests: Entry parsing
# ──────────────────────────────────────────────────────────────


class TestEntryParsing:
    """Test parsing of raw UniProt JSON into UniProtEntry objects."""

    def test_parse_valid_entry(self, mock_uniprot_response: dict) -> None:
        client = UniProtClient()
        raw = mock_uniprot_response["results"][0]
        entry = client._parse_entry(raw)

        assert entry is not None
        assert entry.accession == "P01379"
        assert entry.protein_name == "Cytotoxin 1"
        assert entry.organism == "Naja naja"
        assert entry.organism_id == 8637
        assert entry.sequence_length == 61
        assert "Toxin" in entry.keywords
        assert "Neurotoxin" in entry.keywords
        assert entry.disulfide_bonds == 4
        assert entry.function_annotation is not None
        assert "acetylcholine" in entry.function_annotation.lower()
        assert entry.subcellular_location == "Secreted"
        assert entry.is_reviewed is True

    def test_parse_conotoxin(self, mock_uniprot_response: dict) -> None:
        client = UniProtClient()
        raw = mock_uniprot_response["results"][1]
        entry = client._parse_entry(raw)

        assert entry is not None
        assert entry.accession == "P60301"
        assert entry.protein_name == "Alpha-conotoxin GI"
        assert entry.organism == "Conus geographus"
        assert entry.disulfide_bonds == 2

    def test_parse_fragment_detection(self, fragment_entry_data: dict) -> None:
        client = UniProtClient()
        entry = client._parse_entry(fragment_entry_data)

        assert entry is not None
        assert entry.is_fragment is True

    def test_parse_handles_missing_fields(self) -> None:
        """Parser should handle entries with minimal data gracefully."""
        client = UniProtClient()
        minimal = {
            "primaryAccession": "Q00000",
            "sequence": {"value": "ACDEFGHIKLMNPQRSTVWY", "length": 20},
        }
        entry = client._parse_entry(minimal)

        assert entry is not None
        assert entry.accession == "Q00000"
        assert entry.protein_name == "Unknown"
        assert entry.organism == "Unknown"

    def test_parse_returns_none_on_invalid(self) -> None:
        """Completely broken entries should return None, not raise."""
        client = UniProtClient()
        result = client._parse_entry({"broken": True})
        # Should return None due to missing required fields
        assert result is None


# ──────────────────────────────────────────────────────────────
# Tests: URL construction
# ──────────────────────────────────────────────────────────────


class TestQueryConstruction:
    """Test that API queries are constructed correctly."""

    def test_reviewed_filter_added(self) -> None:
        """When reviewed_only=True, 'AND reviewed:true' should be appended."""
        client = UniProtClient()

        with patch.object(client, "_request_with_retry") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"results": []}
            mock_response.headers = {}
            mock_request.return_value = mock_response

            client.fetch_toxins(query="keyword:KW-0800", reviewed_only=True, max_results=1)

            call_args = mock_request.call_args
            params = call_args[1].get("params") if call_args[1] else call_args[0][1]
            assert "reviewed:true" in params["query"]


# ──────────────────────────────────────────────────────────────
# Tests: Pagination
# ──────────────────────────────────────────────────────────────


class TestPagination:
    """Test cursor-based pagination via Link header."""

    def test_next_url_extracted(self) -> None:
        """Should extract next page URL from Link header."""
        mock_response = MagicMock()
        mock_response.headers = {
            "Link": '<https://rest.uniprot.org/uniprotkb/search?cursor=abc123&size=500>; rel="next"'
        }

        url = UniProtClient._get_next_url(mock_response)
        assert url is not None
        assert "cursor=abc123" in url

    def test_no_next_url_on_last_page(self) -> None:
        """Should return None when there's no Link header."""
        mock_response = MagicMock()
        mock_response.headers = {}

        url = UniProtClient._get_next_url(mock_response)
        assert url is None

    def test_max_results_stops_pagination(self) -> None:
        """Should stop fetching when max_results is reached."""
        client = UniProtClient()
        mock_results = [
            {
                "primaryAccession": f"P{i:05d}",
                "sequence": {"value": "ACDEFGHIKLMNPQRSTVWY", "length": 20},
            }
            for i in range(10)
        ]

        with patch.object(client, "_request_with_retry") as mock_request:
            mock_response = MagicMock()
            mock_response.json.return_value = {"results": mock_results}
            mock_response.headers = {
                "Link": '<https://next.page>; rel="next"'
            }
            mock_request.return_value = mock_response

            entries = client.fetch_toxins(max_results=5)
            assert len(entries) <= 5


# ──────────────────────────────────────────────────────────────
# Tests: Retry logic
# ──────────────────────────────────────────────────────────────


class TestRetryLogic:
    """Test exponential backoff and rate limiting."""

    @patch("venomsearch.etl.uniprot_client.time.sleep")
    def test_retry_on_rate_limit(self, mock_sleep: MagicMock) -> None:
        """Should retry after 429 with Retry-After header."""
        client = UniProtClient()

        rate_limited = MagicMock()
        rate_limited.status_code = 429
        rate_limited.headers = {"Retry-After": "1"}

        success = MagicMock()
        success.status_code = 200
        success.raise_for_status = MagicMock()

        client.session.get = MagicMock(side_effect=[rate_limited, success])

        result = client._request_with_retry("https://test.url")
        assert result == success
        mock_sleep.assert_called_once_with(1)

    @patch("venomsearch.etl.uniprot_client.time.sleep")
    def test_raises_after_max_retries(self, mock_sleep: MagicMock) -> None:
        """Should raise HTTPError after exhausting retries."""
        client = UniProtClient()
        client.session.get = MagicMock(
            side_effect=requests.ConnectionError("Connection failed")
        )

        with pytest.raises(requests.HTTPError, match="Failed after"):
            client._request_with_retry("https://test.url")
