from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional

import typer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from diac.backends.base import get_backend
from diac.data import load_sentences, make_pairs
from diac.metrics import compute_der

app = typer.Typer(help="Arabic diacritization CLI — infer / finetune / evaluate")


@app.command()
def infer(
    model: str = typer.Option(..., help="Backend: camel, byt5, rababa, catt"),
    input: Path = typer.Option(..., help="Input .txt (one sentence per line, undiacritized)"),
    output: Optional[Path] = typer.Option(None, help="Output file (default: stdout)"),
    checkpoint: Optional[Path] = typer.Option(None, help="Path to model checkpoint"),
    batch_size: int = typer.Option(1, help="Batch size passed to backend.infer (backends that don't support it ignore it)"),
):
    backend = get_backend(model)
    _load_checkpoint(backend, model, checkpoint)
    sentences = load_sentences(input)
    results = backend.infer(sentences, batch_size=batch_size)
    text = "\n".join(results)
    if output:
        output.write_text(text + "\n", encoding="utf-8")
    else:
        typer.echo(text)


@app.command()
def finetune(
    model: str = typer.Option(..., help="Backend: byt5, rababa, catt"),
    train: Path = typer.Option(..., help="Training .txt (diacritized, one sentence per line)"),
    dev: Path = typer.Option(..., help="Validation .txt (diacritized, one sentence per line)"),
    output_dir: Path = typer.Option(Path("checkpoints"), help="Where to save checkpoints"),
    epochs: int = typer.Option(3, help="Number of training epochs"),
    batch_size: int = typer.Option(16, help="Batch size"),
):
    backend = get_backend(model)
    train_pairs = make_pairs(load_sentences(train))
    dev_pairs   = make_pairs(load_sentences(dev))
    try:
        backend.finetune(train_pairs, dev_pairs,
                         output_dir=str(output_dir),
                         epochs=epochs,
                         batch_size=batch_size)
    except NotImplementedError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"Fine-tuning complete. Checkpoint saved to {output_dir}")


@app.command()
def evaluate(
    model: str = typer.Option(..., help="Backend: camel, byt5, rababa, catt"),
    input: Path = typer.Option(..., help="Input .txt (undiacritized)"),
    ref: Path = typer.Option(..., help="Reference .txt (diacritized gold)"),
    checkpoint: Optional[Path] = typer.Option(None, help="Path to model checkpoint"),
    format: str = typer.Option("text", help="Output format: text or json"),
):
    if format not in ("text", "json"):
        typer.echo(f"Error: --format must be 'text' or 'json', got '{format}'", err=True)
        raise typer.Exit(code=1)
    backend = get_backend(model)
    _load_checkpoint(backend, model, checkpoint)
    sentences = load_sentences(input)
    hyp = backend.infer(sentences)
    ref_lines = load_sentences(ref)
    if len(hyp) != len(ref_lines):
        typer.echo(
            f"Error: input produced {len(hyp)} hypothesis lines but ref has "
            f"{len(ref_lines)} lines. Check for blank lines in your files.",
            err=True,
        )
        raise typer.Exit(code=1)
    scores = compute_der(hyp, ref_lines)
    if format == "json":
        typer.echo(json.dumps(scores))
    else:
        for k, v in scores.items():
            typer.echo(f"{k}: {v:.4f}")


def _load_checkpoint(backend, model: str, checkpoint: Optional[Path]) -> None:
    """Load checkpoint from --checkpoint flag or env var fallback."""
    _ENV_VARS = {"catt": "CATT_CHECKPOINT", "rababa": "RABABA_CHECKPOINT"}
    path = checkpoint or (
        Path(env_val) if (env_val := os.environ.get(_ENV_VARS.get(model, ""))) else None
    )
    if path:
        backend.load(str(path))
