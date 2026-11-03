"""Command-line entry point: `sec-rag download | parse | index | ask`."""

import logging

import typer

app = typer.Typer(help="RAG over SEC 10-K filings")
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


@app.command()
def download(tickers: list[str], years: int = typer.Option(3, help="Most recent N 10-Ks")):
    """Download 10-K filings from EDGAR for the given tickers."""
    from sec_rag.ingestion.edgar_client import EdgarClient

    client = EdgarClient()
    for ticker in tickers:
        for filing in client.list_10k_filings(ticker, limit=years):
            path = client.download(filing)
            typer.echo(f"{ticker} FY{filing.fiscal_year} -> {path}")


@app.command()
def parse():
    """Parse downloaded filings into sections (data/raw -> data/processed)."""
    from sec_rag.indexing.pipeline import parse_filings

    typer.echo(f"Parsed {parse_filings()} filings")


@app.command()
def index(
    keyword_only: bool = typer.Option(
        False, help="Only rebuild the keyword index from chunks already in the vector store"
    ),
):
    """Chunk, embed and index the parsed filings (run `sec-rag parse` first)."""
    from sec_rag.indexing.pipeline import (
        NoParsedFilingsError,
        index_filings,
        rebuild_keyword_index,
        unparsed_filings,
    )

    if keyword_only:
        typer.echo(f"Keyword-indexed {rebuild_keyword_index()} chunks")
        return
    unparsed = unparsed_filings()
    if unparsed:
        names = ", ".join(f"{m.ticker} FY{m.fiscal_year}" for m in unparsed)
        typer.echo(f"Not parsed yet, skipping: {names}. Run `sec-rag parse` to include them.")
    try:
        typer.echo(f"Indexed {index_filings()} chunks")
    except NoParsedFilingsError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from error


@app.command()
def ask(
    question: str,
    ticker: str | None = None,
    year: int | None = None,
    top_k: int = 8,
):
    """Ask a question over the indexed filings."""
    from sec_rag.pipeline import RAGPipeline

    filters = {k: v for k, v in {"ticker": ticker, "fiscal_year": year}.items() if v}
    result = RAGPipeline().ask_stream(question, filters=filters, top_k=top_k)
    for text in result.text_stream:
        typer.echo(text, nl=False)
    typer.echo("\n\nSources:")
    for i, rc in enumerate(result.sources, start=1):
        c = rc.chunk
        typer.echo(f"  [{i}] {c.ticker} FY{c.fiscal_year} Item {c.item} (relevance {rc.score:.3f})")


if __name__ == "__main__":
    app()
