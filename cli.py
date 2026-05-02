import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).parent))

from src import indexer, rag, vector_store


@click.group()
def cli():
    """doc-assistant — ask questions about your PDFs."""


@cli.command()
@click.argument("path", type=click.Path(exists=True))
def index(path):
    """Index a PDF file into the vector store."""
    result = indexer.index_pdf(path)
    if result["skipped"]:
        click.echo(f"Already indexed (skipping): {Path(path).name}")
    else:
        click.echo(f"Indexed {Path(path).name}: {result['chunks']} chunks across {result['pages']} pages.")


@cli.command()
@click.argument("question")
@click.option("--file", default=None, help="Restrict search to this filename (basename only).")
@click.option("--k", default=4, show_default=True, help="Number of chunks to retrieve.")
def ask(question, file, k):
    """Ask a question across indexed PDFs."""
    result = rag.answer(question, file=file, k=k)
    click.echo(f"\n{result['answer']}\n")
    if result["sources"]:
        click.echo("Sources:")
        for s in result["sources"]:
            click.echo(f"  [{s['file']}, p.{s['page']}]")


@cli.command("list")
def list_files():
    """List all indexed PDF filenames."""
    files = vector_store.list_indexed_files()
    if not files:
        click.echo("No PDFs indexed yet.")
    else:
        click.echo("Indexed files:")
        for f in files:
            click.echo(f"  {f}")


@cli.command()
@click.confirmation_option(prompt="This will delete all indexed data. Are you sure?")
def clear():
    """Remove all indexed data from the vector store."""
    vector_store.clear_all()
    click.echo("Vector store cleared.")


if __name__ == "__main__":
    cli()
