<div align="center">

# 🐍 VenomSearch-AI

**ETL Pipeline & Vector Search for Neurotoxins using ESM-2 Protein Language Model**

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)]()
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

*Bioinformatics & AI Data Engineering portfolio project demonstrating biological API consumption, protein sequence processing, pLLM inference, and vector indexing for in silico screening.*

</div>

---

## 🏗️ Architecture

```
UniProt API (Tox-Prot)
       │ (REST / Streaming JSON)
       ▼
┌─────────────────────────────┐
│  1. Ingestion & Validation  │──▶ Pydantic v2 + Polars LazyFrames
│     (ETL Pipeline)          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  2. ESM-2 Inference Engine  │──▶ Meta ESM-2 (esm2_t6_8M_UR50D)
│     (Protein Embeddings)    │    Mean pooling → dim 320
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  3. Storage & Indexing      │──▶ LanceDB (Vector) + Parquet (Cold)
│     (Dual Storage)          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│  4. Query CLI               │──▶ Cosine Similarity Search + Ranking
│     (Typer + Rich)          │
└─────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/venomsearch-ai.git
cd venomsearch-ai

# Install dependencies with uv
uv sync

# Install with notebook extras (for UMAP visualization)
uv sync --extra notebook
```

### Usage

```bash
# 1. Ingest toxin data from UniProt/Tox-Prot
venomsearch ingest

# 2. Generate ESM-2 embeddings
venomsearch embed

# 3. Build vector index
venomsearch index

# 4. Search for similar toxins
venomsearch search "MKTLLLTLVVVTIVCLDLGYTRDCIRFHDKCSIHRECMQCCRSIGYVHVFRKRN"

# 5. Search from a FASTA file
venomsearch search --file query.fasta

# 6. Dataset statistics
venomsearch info

# 7. Run CPU vs GPU benchmark
venomsearch benchmark
```

### Docker

```bash
docker build -t venomsearch-ai .
docker run venomsearch-ai search "MKTLLLTLVVVTIVCLDLGYT..."
```

## 📊 Example Output

```
╭─────────────────────────────────────────────────────────────╮
│  🔬 VenomSearch-AI — Top 5 Similar Toxins                  │
├──────┬─────────────┬──────────────┬────────────┬────────────┤
│ Rank │ Accession   │ Protein Name │ Organism   │ Cos. Sim.  │
├──────┼─────────────┼──────────────┼────────────┼────────────┤
│  1   │ P01437      │ 3FTx-Naja    │ N. naja    │ 0.9842     │
│  2   │ P60301      │ α-conotoxin  │ C. geogr.  │ 0.9713     │
│  3   │ P0C1Z0      │ δ-SVNTX      │ B. jara.   │ 0.9651     │
│  4   │ Q9TWG0      │ κ-SNTX       │ S. invicta │ 0.9488     │
│  5   │ P0DL46      │ μ-conotoxin  │ C. magus   │ 0.9301     │
╰──────┴─────────────┴──────────────┴────────────┴────────────╯
```

## 🧬 Technical Stack

| Layer | Technology | Purpose |
|:------|:-----------|:--------|
| **Validation** | Pydantic v2 | Typed API response parsing |
| **DataFrame** | Polars | Lazy ETL with Rust backend |
| **pLLM** | ESM-2 (8M params) | Protein sequence embeddings |
| **Vector DB** | LanceDB | Embedded cosine similarity search |
| **Cold Storage** | Apache Parquet | Columnar analytics storage |
| **CLI** | Typer + Rich | Interactive command-line interface |
| **Visualization** | UMAP + Plotly | Latent space exploration |

## ⚡ Benchmark

| Metric | CPU | GPU |
|:-------|:----|:----|
| 100 sequences (avg 80 aa) | — | — |
| 1000 sequences (avg 80 aa) | — | — |
| Peak Memory | — | — |
| Index Creation (LanceDB) | — | — |
| Search Latency (top-5) | — | — |

*Run `venomsearch benchmark` to fill in your hardware-specific numbers.*

## 📁 Project Structure

```
venomsearch-ai/
├── data/
│   ├── raw/                    # UniProt JSONL snapshots
│   └── processed/              # Parquet with embeddings & metadata
├── src/venomsearch/
│   ├── etl/
│   │   ├── uniprot_client.py   # Paginated UniProt API client
│   │   ├── cleaner.py          # FASTA sequence validation
│   │   └── pipeline.py         # ETL orchestrator
│   ├── embeddings/
│   │   ├── esm_encoder.py      # ESM-2 inference with PyTorch
│   │   └── batch_sampler.py    # Dynamic padding & batching
│   ├── search/
│   │   ├── vector_store.py     # LanceDB interface
│   │   └── similarity.py       # Ranking & search engine
│   ├── models.py               # Pydantic schemas
│   └── cli.py                  # Typer CLI application
├── notebooks/
│   └── 01_eda_and_umap.ipynb   # UMAP latent space visualization
├── tests/                      # Pytest test suite
├── Dockerfile                  # Reproducible container
├── pyproject.toml              # Project configuration
└── README.md
```

## 🔬 Data Source

This project uses data from **UniProt Tox-Prot**, the toxin annotation program of the Universal Protein Resource. Specifically:

- **KW-0800** — Toxin (general category)
- **KW-0528** — Neurotoxin
- **KW-0123** — Cardiotoxin
- **KW-0008** — Acetylcholine receptor inhibiting toxin

Only **reviewed (Swiss-Prot)** entries are used to ensure high-quality curated annotations.

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

## 📚 References

- Rives, A. et al. (2021). *Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences.* PNAS.
- Lin, Z. et al. (2023). *Evolutionary-scale prediction of atomic-level protein structure with a language model.* Science.
- [UniProt Tox-Prot](https://www.uniprot.org/help/Toxin_annotation_program)
- [LanceDB Documentation](https://lancedb.github.io/lancedb/)
