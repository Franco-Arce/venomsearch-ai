"""VenomSearch-AI Command-Line Interface.

Professional CLI built with Typer + Rich for interacting with the
toxin ETL pipeline, embedding engine, and vector search.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="venomsearch",
    help="🐍 VenomSearch-AI — ETL Pipeline & Vector Search for Neurotoxins with ESM-2",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()

# ──────────────────────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────────────────────


def _setup_logging(verbose: bool = False) -> None:
    """Configure rich logging handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


# ──────────────────────────────────────────────────────────────
# Commands
# ──────────────────────────────────────────────────────────────


@app.command()
def ingest(
    data_dir: str = typer.Option("data", help="Root data directory for output"),
    reviewed_only: bool = typer.Option(True, help="Only fetch Swiss-Prot (reviewed) entries"),
    max_per_category: int | None = typer.Option(
        None, help="Max entries per toxin category (for testing)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging"),
) -> None:
    """🔌 Ingest toxin data from UniProt/Tox-Prot API.

    Downloads, validates, and cleans protein sequences from all Tox-Prot
    categories. Saves processed data as Parquet + raw JSONL snapshot.
    """
    _setup_logging(verbose)

    from venomsearch.etl.pipeline import ETLPipeline

    console.print(
        Panel(
            "[bold cyan]VenomSearch-AI[/] — ETL Ingestion Pipeline\n"
            f"Data directory: [green]{data_dir}[/]\n"
            f"Reviewed only: {reviewed_only}\n"
            f"Max per category: {max_per_category or 'unlimited'}",
            title="🔌 Ingest",
        )
    )

    pipeline = ETLPipeline(data_dir=Path(data_dir))

    with console.status("[bold green]Fetching from UniProt...", spinner="dots"):
        _df, stats = pipeline.run(
            reviewed_only=reviewed_only,
            max_per_category=max_per_category,
        )

    # Display results
    _display_ingestion_stats(stats)


@app.command()
def embed(
    data_dir: str = typer.Option("data", help="Root data directory"),
    model_name: str = typer.Option(
        "facebook/esm2_t6_8M_UR50D",
        help="ESM-2 model name (8M=t6, 35M=t12, 150M=t30)",
    ),
    batch_size: int = typer.Option(32, help="Sequences per batch"),
    device: str = typer.Option("auto", help="Compute device (auto/cpu/cuda/mps)"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🧬 Generate ESM-2 embeddings for all processed sequences.

    Reads the cleaned Parquet file, computes protein embeddings via
    ESM-2, and saves the result back to Parquet with embedding vectors.
    """
    _setup_logging(verbose)

    import numpy as np
    import polars as pl

    from venomsearch.embeddings.esm_encoder import ESMEncoder

    processed_path = Path(data_dir) / "processed" / "toxins.parquet"
    if not processed_path.exists():
        console.print(
            "[bold red]Error:[/] No processed data found. Run 'venomsearch ingest' first."
        )
        raise typer.Exit(code=1)

    df = pl.read_parquet(processed_path)
    sequences = df["sequence"].to_list()

    console.print(
        Panel(
            f"[bold cyan]ESM-2 Embedding Engine[/]\n"
            f"Model: [green]{model_name}[/]\n"
            f"Device: [green]{device}[/]\n"
            f"Sequences: [yellow]{len(sequences)}[/]\n"
            f"Batch size: {batch_size}",
            title="🧬 Embed",
        )
    )

    encoder = ESMEncoder(model_name=model_name, device=device)

    start = time.perf_counter()
    embeddings = encoder.encode_batch(sequences, batch_size=batch_size)
    elapsed = time.perf_counter() - start

    # Save embeddings alongside metadata
    embeddings_path = Path(data_dir) / "processed" / "embeddings.npy"
    np.save(embeddings_path, embeddings)

    console.print("\n[bold green]✓[/] Embeddings generated successfully!")
    console.print(f"  Shape: [cyan]{embeddings.shape}[/]")
    console.print(f"  Time: [yellow]{elapsed:.1f}s[/] ({len(sequences) / elapsed:.1f} seqs/s)")
    console.print(f"  Saved: [dim]{embeddings_path}[/]")


@app.command()
def index(
    data_dir: str = typer.Option("data", help="Root data directory"),
    db_path: str = typer.Option("data/lancedb", help="LanceDB database path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🗄️ Build/rebuild the LanceDB vector index.

    Reads processed Parquet + embeddings and creates the searchable
    vector index for cosine similarity queries.
    """
    _setup_logging(verbose)

    import numpy as np
    import polars as pl

    from venomsearch.search.vector_store import VenomVectorStore

    processed_path = Path(data_dir) / "processed" / "toxins.parquet"
    embeddings_path = Path(data_dir) / "processed" / "embeddings.npy"

    if not processed_path.exists() or not embeddings_path.exists():
        console.print(
            "[bold red]Error:[/] Missing processed data or embeddings. "
            "Run 'venomsearch ingest' and 'venomsearch embed' first."
        )
        raise typer.Exit(code=1)

    df = pl.read_parquet(processed_path)
    embeddings = np.load(embeddings_path)

    console.print(
        Panel(
            f"[bold cyan]Vector Index Builder[/]\n"
            f"Entries: [yellow]{len(df)}[/]\n"
            f"Embedding dim: [green]{embeddings.shape[1]}[/]\n"
            f"DB path: [dim]{db_path}[/]",
            title="🗄️ Index",
        )
    )

    store = VenomVectorStore(db_path=db_path)

    start = time.perf_counter()
    store.create_index(df, embeddings, overwrite=True)
    elapsed = time.perf_counter() - start

    console.print(f"\n[bold green]✓[/] Index created in {elapsed:.2f}s")

    # Display index info
    info = store.get_table_info()
    console.print(f"  Total indexed: [cyan]{info.get('total_rows', 0)}[/] toxins")
    console.print(f"  Embedding dim: [cyan]{info.get('embedding_dim', 0)}[/]")


@app.command()
def search(
    sequence: str | None = typer.Argument(
        None, help="Amino acid sequence to search (one-letter code)"
    ),
    file: str | None = typer.Option(None, "--file", "-f", help="Path to FASTA file"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of results"),
    model_name: str = typer.Option(
        "facebook/esm2_t6_8M_UR50D", "--model", "-m", help="ESM-2 model"
    ),
    db_path: str = typer.Option("data/lancedb", help="LanceDB path"),
    device: str = typer.Option("auto", help="Compute device"),
    family: str | None = typer.Option(None, help="Filter by toxin family"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """🔬 Search for similar toxins given a peptide sequence.

    Computes the ESM-2 embedding of a query sequence on the fly and
    returns the top-k most similar toxins from the indexed Tox-Prot database.
    """
    _setup_logging(verbose)

    from venomsearch.search.similarity import ToxinSearchEngine

    if not sequence and not file:
        console.print("[bold red]Error:[/] Provide a sequence or --file path.")
        raise typer.Exit(code=1)

    console.print(
        Panel("[bold cyan]VenomSearch-AI[/] — Similarity Search", title="🔬 Search")
    )

    engine = ToxinSearchEngine.from_paths(
        model_name=model_name,
        db_path=db_path,
        device=device,
    )

    if file:
        responses = engine.search_fasta_file(file, top_k=top_k, filter_family=family)
        for i, response in enumerate(responses):
            console.print(f"\n[bold]Query {i + 1}:[/] {response.query_sequence[:50]}...")
            _display_search_results(response)
    else:
        assert sequence is not None
        response = engine.search(
            query_sequence=sequence,
            top_k=top_k,
            filter_family=family,
        )
        _display_search_results(response)


@app.command()
def info(
    data_dir: str = typer.Option("data", help="Root data directory"),
    db_path: str = typer.Option("data/lancedb", help="LanceDB path"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """📊 Display dataset and index statistics."""
    _setup_logging(verbose)

    import polars as pl

    from venomsearch.search.vector_store import VenomVectorStore

    console.print(
        Panel("[bold cyan]VenomSearch-AI[/] — Dataset Info", title="📊 Info")
    )

    # Dataset stats
    processed_path = Path(data_dir) / "processed" / "toxins.parquet"
    if processed_path.exists():
        df = pl.read_parquet(processed_path)

        table = Table(title="Dataset Overview", show_lines=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Total entries", str(len(df)))
        table.add_row("Unique organisms", str(df["organism"].n_unique()))
        table.add_row("Avg sequence length", f'{df["sequence_length"].mean():.1f} aa')
        table.add_row("Min sequence length", f'{df["sequence_length"].min()} aa')
        table.add_row("Max sequence length", f'{df["sequence_length"].max()} aa')
        console.print(table)

        # Family distribution
        family_table = Table(title="Toxin Family Distribution", show_lines=True)
        family_table.add_column("Family", style="cyan")
        family_table.add_column("Count", style="yellow", justify="right")
        family_table.add_column("Percentage", style="green", justify="right")

        family_counts = (
            df.group_by("toxin_family")
            .len()
            .sort("len", descending=True)
        )
        for row in family_counts.iter_rows(named=True):
            pct = row["len"] / len(df) * 100
            family_table.add_row(row["toxin_family"], str(row["len"]), f"{pct:.1f}%")

        console.print(family_table)
    else:
        console.print("[yellow]No processed dataset found. Run 'venomsearch ingest' first.[/]")

    # Index stats
    try:
        store = VenomVectorStore(db_path=db_path)
        index_info = store.get_table_info()
        if index_info.get("exists"):
            console.print(
                f"\n[bold]Vector Index:[/] [green]✓ Active[/] — "
                f"{index_info['total_rows']} entries, "
                f"dim={index_info['embedding_dim']}"
            )
        else:
            console.print("\n[bold]Vector Index:[/] [yellow]Not created[/]")
    except Exception:
        console.print("\n[bold]Vector Index:[/] [yellow]Not available[/]")


@app.command()
def benchmark(
    data_dir: str = typer.Option("data", help="Root data directory"),
    n_sequences: int = typer.Option(100, help="Number of sequences to benchmark"),
    model_name: str = typer.Option("facebook/esm2_t6_8M_UR50D", help="ESM-2 model"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """⚡ Run embedding performance benchmark.

    Measures inference speed, memory usage, and search latency
    across different batch sizes.
    """
    _setup_logging(verbose)

    import polars as pl

    from venomsearch.embeddings.esm_encoder import ESMEncoder

    processed_path = Path(data_dir) / "processed" / "toxins.parquet"
    if not processed_path.exists():
        console.print("[bold red]Error:[/] No data. Run 'venomsearch ingest' first.")
        raise typer.Exit(code=1)

    df = pl.read_parquet(processed_path)
    sequences = df["sequence"].to_list()[:n_sequences]

    console.print(
        Panel(
            f"[bold cyan]Benchmark[/]\n"
            f"Model: [green]{model_name}[/]\n"
            f"Sequences: [yellow]{len(sequences)}[/]",
            title="⚡ Benchmark",
        )
    )

    # Test different batch sizes
    batch_sizes = [8, 16, 32, 64]
    results_table = Table(title="Embedding Benchmark", show_lines=True)
    results_table.add_column("Batch Size", style="cyan", justify="center")
    results_table.add_column("Total Time (s)", style="yellow", justify="right")
    results_table.add_column("Seqs/sec", style="green", justify="right")

    encoder = ESMEncoder(model_name=model_name, device="auto")

    for bs in batch_sizes:
        if bs > len(sequences):
            continue
        start = time.perf_counter()
        _ = encoder.encode_batch(sequences, batch_size=bs, show_progress=False)
        elapsed = time.perf_counter() - start
        seqs_per_sec = len(sequences) / elapsed

        results_table.add_row(str(bs), f"{elapsed:.2f}", f"{seqs_per_sec:.1f}")

    console.print(results_table)

    # Model info
    info_table = Table(title="Model Info", show_lines=True)
    info_table.add_column("Property", style="cyan")
    info_table.add_column("Value", style="green")
    for key, value in encoder.model_info.items():
        info_table.add_row(key, str(value))
    console.print(info_table)


# ──────────────────────────────────────────────────────────────
# Display helpers
# ──────────────────────────────────────────────────────────────


def _display_search_results(response: list[dict[str, Any]]) -> None:
    """Render search results as a rich table."""

    table = Table(
        title=f"🔬 VenomSearch-AI — Top {len(response.results)} Similar Toxins",
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Rank", style="bold", justify="center", width=4)
    table.add_column("Accession", style="cyan", width=10)
    table.add_column("Protein Name", style="white", max_width=30)
    table.add_column("Organism", style="green", max_width=20)
    table.add_column("Family", style="yellow", width=14)
    table.add_column("Cos. Sim.", style="bold magenta", justify="right", width=9)

    for r in response.results:
        # Color-code similarity score
        score = r.cosine_similarity
        if score >= 0.95:
            score_style = "[bold green]"
        elif score >= 0.85:
            score_style = "[yellow]"
        else:
            score_style = "[red]"

        table.add_row(
            str(r.rank),
            r.accession,
            r.protein_name[:30],
            r.organism[:20],
            r.toxin_family,
            f"{score_style}{score:.4f}[/]",
        )

    console.print(table)
    console.print(
        f"  [dim]Query: {response.query_sequence[:60]}... ({response.query_length} aa)[/]"
    )
    console.print(
        f"  [dim]Indexed: {response.total_indexed} toxins | "
        f"Search time: {response.search_time_ms:.1f}ms[/]"
    )


def _display_ingestion_stats(stats: dict[str, int]) -> None:
    """Render ingestion statistics as rich tables."""

    table = Table(title="ETL Ingestion Results", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Total fetched from UniProt", str(stats.total_fetched))
    table.add_row("After cleaning", str(stats.total_after_cleaning))
    table.add_row("Unique organisms", str(stats.organisms_count))
    table.add_row("Avg sequence length", f"{stats.avg_sequence_length:.1f} aa")

    console.print(table)

    if stats.families:
        family_table = Table(title="Toxin Families", show_lines=True)
        family_table.add_column("Family", style="cyan")
        family_table.add_column("Count", style="yellow", justify="right")

        for family, count in sorted(stats.families.items(), key=lambda x: x[1], reverse=True):
            family_table.add_row(family, str(count))

        console.print(family_table)


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()
