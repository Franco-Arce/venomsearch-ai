"""Shared test fixtures for VenomSearch-AI test suite.

Provides sample sequences, mock API responses, and pre-computed
embeddings for consistent, fast testing without external dependencies.
"""

from __future__ import annotations

import numpy as np
import pytest

# ──────────────────────────────────────────────────────────────
# Sample protein sequences (real toxins from Tox-Prot)
# ──────────────────────────────────────────────────────────────

#: a-cobratoxin from Naja kaouthia (~71 aa, neurotoxin)
COBRATOXIN_SEQ = (
    "MKTLLLTLVVVTIVCLDLGYTIRCFITPDITSKDCPNGHVCYTKTWCDAFCSIR"
    "GKRVDLGCAATCPTVKTGVD"
)

#: μ-conotoxin GIIIA from Conus geographus (~22 aa, Na+ channel blocker)
CONOTOXIN_SEQ = "RDCCTOOKKCKDRQCKOQRCCA"

#: Melittin from Apis mellifera (honeybee, ~26 aa, cytolytic)
MELITTIN_SEQ = "GIGAVLKVLTTGLPALISWIKRKRQQ"

#: Chlorotoxin from Leiurus quinquestriatus (~36 aa, Cl- channel blocker)
CHLOROTOXIN_SEQ = "MCMPCFTTDHQMARKCDDCCGGKGRGKCYGPQCLCR"

#: Short synthetic peptide for edge case testing
SHORT_PEPTIDE_SEQ = "ACDEFGHIKLMNPQRSTVWY"


@pytest.fixture
def sample_sequences() -> list[str]:
    """List of 5 real toxin sequences for testing."""
    return [
        COBRATOXIN_SEQ,
        CONOTOXIN_SEQ,
        MELITTIN_SEQ,
        CHLOROTOXIN_SEQ,
        SHORT_PEPTIDE_SEQ,
    ]


@pytest.fixture
def single_sequence() -> str:
    """A single toxin sequence (cobratoxin)."""
    return COBRATOXIN_SEQ


@pytest.fixture
def ambiguous_sequences() -> list[str]:
    """Sequences with non-canonical residues that should be rejected."""
    return [
        "ACDEFXHIKLM",       # Contains X (unknown)
        "ACDEFBHIKLM",       # Contains B (Asx = Asp or Asn)
        "ACDEFZHIKLM",       # Contains Z (Glx = Glu or Gln)
        "ACDEFJHIKLM",       # Contains J (Leu or Ile)
        "ACDEFULHIKLM",      # Contains U (selenocysteine — may be allowed)
        "ACDEFOHIKLM",       # Contains O (pyrrolysine)
    ]


@pytest.fixture
def fragment_entry_data() -> dict:
    """Mock UniProt entry data for a protein fragment."""
    return {
        "primaryAccession": "P99999",
        "uniProtkbId": "FRAG_TEST",
        "proteinDescription": {
            "recommendedName": {"fullName": {"value": "Test Fragment"}},
            "flag": "Fragment",
        },
        "organism": {"scientificName": "Test organism", "taxonId": 9999},
        "sequence": {"value": "ACDEFGHIKLMNPQRSTVWY", "length": 20},
        "keywords": [{"name": "Toxin", "id": "KW-0800"}],
        "uniProtKBCrossReferences": [],
        "features": [],
        "comments": [],
        "entryType": "UniProtKB reviewed (Swiss-Prot)",
    }


@pytest.fixture
def mock_uniprot_response() -> dict:
    """Mock JSON response from UniProt API search endpoint."""
    return {
        "results": [
            {
                "primaryAccession": "P01379",
                "uniProtkbId": "3SA1_NAJNA",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Cytotoxin 1"}
                    }
                },
                "organism": {"scientificName": "Naja naja", "taxonId": 8637},
                "sequence": {
                    "value": "LECHNQQSSQTPTTTGCSGGETNCYKKRWRDHRGYRTERGCGCPSVKNGIEINCCTTDRCNN",
                    "length": 61,
                },
                "keywords": [
                    {"name": "Toxin", "id": "KW-0800"},
                    {"name": "Neurotoxin", "id": "KW-0528"},
                ],
                "uniProtKBCrossReferences": [
                    {"database": "GO", "id": "GO:0005576"},
                    {"database": "GO", "id": "GO:0090729"},
                ],
                "features": [
                    {"type": "Disulfide bond"},
                    {"type": "Disulfide bond"},
                    {"type": "Disulfide bond"},
                    {"type": "Disulfide bond"},
                ],
                "comments": [
                    {
                        "commentType": "FUNCTION",
                        "texts": [{"value": "Binds to nicotinic acetylcholine receptor."}],
                    },
                    {
                        "commentType": "SUBCELLULAR LOCATION",
                        "subcellularLocations": [
                            {"location": {"value": "Secreted"}}
                        ],
                    },
                ],
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
            },
            {
                "primaryAccession": "P60301",
                "uniProtkbId": "CA13_CONGE",
                "proteinDescription": {
                    "recommendedName": {
                        "fullName": {"value": "Alpha-conotoxin GI"}
                    }
                },
                "organism": {
                    "scientificName": "Conus geographus",
                    "taxonId": 6491,
                },
                "sequence": {"value": "ECCNPACGRHYSC", "length": 13},
                "keywords": [
                    {"name": "Toxin", "id": "KW-0800"},
                    {"name": "Neurotoxin", "id": "KW-0528"},
                    {"name": "Acetylcholine receptor inhibiting toxin", "id": "KW-0008"},
                ],
                "uniProtKBCrossReferences": [],
                "features": [
                    {"type": "Disulfide bond"},
                    {"type": "Disulfide bond"},
                ],
                "comments": [],
                "entryType": "UniProtKB reviewed (Swiss-Prot)",
            },
        ]
    }


@pytest.fixture
def sample_embeddings() -> np.ndarray:
    """Pre-generated random embeddings for 5 sequences (dim=320)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((5, 320)).astype(np.float32)
