# diac-ft CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a unified Python CLI (`diac infer/finetune/evaluate`) over four Arabic diacritization backends: CAMeL MLE, Rababa, ByT5/Fine-Tashkeel, and CATT.

**Architecture:** Adapter pattern — a `DiacritizationBackend` ABC defines `infer/finetune/save/load`. Each backend is a concrete subclass. `data.py` and `metrics.py` are shared utilities. Typer wires the three CLI subcommands to the registry.

**Tech Stack:** Python 3.11+, PyTorch 2.4+, transformers 4.46+, camel-tools 1.6+, diacritization-evaluation 0.5+, typer 0.12+, omegaconf 2.3+, pytest.

## Global Constraints

- Python >= 3.11 everywhere.
- No TensorFlow dependency anywhere.
- `camel-tools` must be pip-installed (not imported from `codes/`). Requires `brew install cmake boost` on macOS.
- `codes/rababa/` is installed as a local editable package: `pip install -e codes/rababa/`.
- CATT requires cloning `github.com/abjadai/catt` into `codes/catt/` and installing: `pip install -e codes/catt/`.
- `diacritization-evaluation` DER functions take **file paths**: `der.calculate_der_from_path(orig_path, pred_path)`.
- All CLI commands accept `--model {camel,rababa,byt5,catt}`.
- `camel` backend: inference only — `finetune()` raises `NotImplementedError`.
- Licence: Apache 2.0.

---

## File Map

| File | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, `[project.scripts]` entry point |
| `LICENSE` | Apache 2.0 text |
| `diac/__init__.py` | Empty package marker |
| `diac/data.py` | `load_sentences(path)`, `make_pairs(sentences)` |
| `diac/metrics.py` | `compute_der(hyp, ref)` → `{"DER", "DER*", "WER", "WER*"}` |
| `diac/backends/base.py` | `DiacritizationBackend` ABC + `get_backend(name)` registry |
| `diac/backends/camel.py` | `CAMeLBackend` — MLE morphology inference |
| `diac/backends/byt5.py` | `ByT5Backend` — HF Seq2SeqTrainer infer + finetune |
| `diac/backends/rababa.py` | `RababaBackend` — Rababa modern infer + train_supervised |
| `diac/backends/catt.py` | `CATTBackend` — abjadai/catt char-BERT infer + finetune |
| `diac/cli.py` | Typer app: `infer`, `finetune`, `evaluate` subcommands |
| `tests/test_data.py` | Unit tests for data.py |
| `tests/test_metrics.py` | Unit tests for metrics.py |
| `tests/test_backends_camel.py` | Integration test for CAMeLBackend |
| `tests/test_cli.py` | CLI smoke tests via `typer.testing.CliRunner` |

---

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `LICENSE`
- Create: `diac/__init__.py`
- Create: `diac/backends/__init__.py`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: `diac` package importable; `diac` CLI entry point registered.

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "diac-ft"
version = "0.1.0"
requires-python = ">=3.11"
license = { text = "Apache-2.0" }
dependencies = [
    "torch>=2.4",
    "transformers>=4.46",
    "datasets>=3.0",
    "camel-tools>=1.6",
    "diacritization-evaluation>=0.5",
    "typer>=0.12",
    "omegaconf>=2.3",
]

[project.scripts]
diac = "diac.cli:app"

[project.optional-dependencies]
dev = ["pytest>=8", "pytest-cov"]
```

- [ ] **Step 2: Create LICENSE**

Download Apache 2.0 text from https://www.apache.org/licenses/LICENSE-2.0.txt and save as `LICENSE`. Replace `[yyyy]` with `2026` and `[name of copyright owner]` with your name.

- [ ] **Step 3: Create package markers**

```bash
touch diac/__init__.py diac/backends/__init__.py tests/__init__.py
```

- [ ] **Step 4: Install in editable mode with dev extras**

```bash
# macOS prerequisite (skip if already done)
brew install cmake boost

# install camel-tools (needs Rust: https://rustup.rs if not present)
pip install camel-tools

# install Rababa from local clone
pip install -e codes/rababa/

# install CATT from local clone (clone first if not done)
# git clone https://github.com/abjadai/catt codes/catt
pip install -e codes/catt/

# install this package in editable mode
pip install -e ".[dev]"
```

- [ ] **Step 5: Verify install**

```bash
python -c "import diac; print('ok')"
diac --help
```

Expected: `ok` printed; `diac --help` shows Typer usage.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml LICENSE diac/ tests/
git commit -m "feat: scaffold diac-ft package"
```

---

## Task 2: data.py — sentence loader and pair maker

**Files:**
- Create: `diac/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Produces:
  - `load_sentences(path: str | Path) -> list[str]` — reads one diacritized sentence per line, strips trailing whitespace, skips blanks.
  - `make_pairs(sentences: list[str]) -> list[tuple[str, str]]` — returns `(undiacritized, diacritized)` pairs using `dediac_ar`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data.py
import tempfile, os
from pathlib import Path
from diac.data import load_sentences, make_pairs

SAMPLE = "فَإِنْ لَمْ يَكُونَا كَذَلِكَ\nقَالَ الْإِسْنَوِيُّ\n\n"

def test_load_sentences_strips_blanks(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    sents = load_sentences(f)
    assert len(sents) == 2
    assert sents[0] == "فَإِنْ لَمْ يَكُونَا كَذَلِكَ"

def test_make_pairs_strips_diacritics():
    sents = ["فَإِنْ"]
    pairs = make_pairs(sents)
    assert len(pairs) == 1
    undiac, diac = pairs[0]
    assert diac == "فَإِنْ"
    assert "َ" not in undiac  # fatha stripped
    assert "فإن" in undiac
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ImportError` (module not yet created).

- [ ] **Step 3: Implement data.py**

```python
# diac/data.py
from __future__ import annotations
from pathlib import Path
from camel_tools.utils.dediac import dediac_ar


def load_sentences(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def make_pairs(sentences: list[str]) -> list[tuple[str, str]]:
    return [(dediac_ar(s), s) for s in sentences]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diac/data.py tests/test_data.py
git commit -m "feat: add data loader with dediac_ar pair extraction"
```

---

## Task 3: metrics.py — DER/WER wrapper

**Files:**
- Create: `diac/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `from diacritization_evaluation import der, wer`
- Produces:
  - `compute_der(hyp: list[str], ref: list[str]) -> dict[str, float]`
    Returns `{"DER": float, "DER*": float, "WER": float, "WER*": float}`.
    `DER*`/`WER*` are case-ending-insensitive variants (standard in Arabic diacritization evaluation).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
from diac.metrics import compute_der

PERFECT_HYP = ["فَإِنْ لَمْ"]
PERFECT_REF = ["فَإِنْ لَمْ"]
WRONG_HYP   = ["فإن لم"]   # no diacritics at all

def test_perfect_prediction_gives_zero_der():
    result = compute_der(PERFECT_HYP, PERFECT_REF)
    assert result["DER"] == 0.0
    assert result["WER"] == 0.0

def test_missing_diacritics_gives_nonzero_der():
    result = compute_der(WRONG_HYP, PERFECT_REF)
    assert result["DER"] > 0.0
    assert result["WER"] > 0.0

def test_returns_all_keys():
    result = compute_der(PERFECT_HYP, PERFECT_REF)
    assert set(result.keys()) == {"DER", "DER*", "WER", "WER*"}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement metrics.py**

```python
# diac/metrics.py
from __future__ import annotations
import tempfile
from pathlib import Path
from diacritization_evaluation import der, wer


def compute_der(hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Compute DER and WER between hypothesis and reference sentence lists.

    Writes temporary files because diacritization_evaluation operates on paths.
    """
    with tempfile.TemporaryDirectory() as tmp:
        orig_path = str(Path(tmp) / "ref.txt")
        pred_path = str(Path(tmp) / "hyp.txt")
        Path(orig_path).write_text("\n".join(ref), encoding="utf-8")
        Path(pred_path).write_text("\n".join(hyp), encoding="utf-8")
        return {
            "DER":  der.calculate_der_from_path(orig_path, pred_path),
            "DER*": der.calculate_der_from_path(orig_path, pred_path, case_ending=False),
            "WER":  wer.calculate_wer_from_path(orig_path, pred_path),
            "WER*": wer.calculate_wer_from_path(orig_path, pred_path, case_ending=False),
        }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diac/metrics.py tests/test_metrics.py
git commit -m "feat: add DER/WER metrics wrapper"
```

---

## Task 4: backends/base.py — ABC and registry

**Files:**
- Create: `diac/backends/base.py`

**Interfaces:**
- Produces:
  - `class DiacritizationBackend(ABC)` with:
    - `@abstractmethod infer(self, sentences: list[str]) -> list[str]`
    - `finetune(self, train: list[tuple[str,str]], dev: list[tuple[str,str]], **kwargs) -> None` — raises `NotImplementedError` by default
    - `save(self, path: str) -> None` — raises `NotImplementedError` by default
    - `load(self, path: str) -> None` — raises `NotImplementedError` by default
  - `get_backend(name: str) -> DiacritizationBackend` — factory; raises `ValueError` for unknown names

- [ ] **Step 1: Implement base.py**

```python
# diac/backends/base.py
from __future__ import annotations
from abc import ABC, abstractmethod


class DiacritizationBackend(ABC):
    @abstractmethod
    def infer(self, sentences: list[str]) -> list[str]: ...

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        **kwargs,
    ) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning")

    def save(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path: str) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")


_REGISTRY: dict[str, str] = {
    "camel":  "diac.backends.camel.CAMeLBackend",
    "byt5":   "diac.backends.byt5.ByT5Backend",
    "rababa": "diac.backends.rababa.RababaBackend",
    "catt":   "diac.backends.catt.CATTBackend",
}


def get_backend(name: str) -> DiacritizationBackend:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {list(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
```

- [ ] **Step 2: Quick smoke test (no pytest file needed — verify interactively)**

```bash
python -c "
from diac.backends.base import DiacritizationBackend, get_backend
import inspect
print('ABC methods:', [m for m in dir(DiacritizationBackend) if not m.startswith('_')])
"
```

Expected: prints `['finetune', 'infer', 'load', 'save']`.

- [ ] **Step 3: Commit**

```bash
git add diac/backends/base.py
git commit -m "feat: add DiacritizationBackend ABC and registry"
```

---

## Task 5: backends/camel.py — CAMeL MLE inference adapter

**Files:**
- Create: `diac/backends/camel.py`
- Create: `tests/test_backends_camel.py`

**Interfaces:**
- Consumes: `DiacritizationBackend` from `diac.backends.base`
- Produces: `class CAMeLBackend(DiacritizationBackend)` with working `infer(sentences)`

**Note:** CAMeL's MLE disambiguator needs camel_data installed. Run `camel_data -i light` once after installing camel-tools.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends_camel.py
import pytest
from diac.backends.camel import CAMeLBackend

@pytest.fixture(scope="module")
def backend():
    return CAMeLBackend()

def test_infer_returns_same_length(backend):
    sentences = ["فإن لم يكونا", "قال الإسنوي"]
    results = backend.infer(sentences)
    assert len(results) == 2

def test_infer_adds_diacritics(backend):
    result = backend.infer(["كتب"])[0]
    # result should contain at least one diacritic character
    arabic_diacritics = set("ًٌٍَُِّْ")
    assert any(c in arabic_diacritics for c in result)

def test_finetune_raises(backend):
    with pytest.raises(NotImplementedError):
        backend.finetune([], [])
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_backends_camel.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement camel.py**

```python
# diac/backends/camel.py
from __future__ import annotations
import re
from diac.backends.base import DiacritizationBackend
from camel_tools.morphology.database import MorphologyDB
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tokenizers.word import simple_word_tokenize

_WHITESPACE_RE = re.compile(r'\s+|\S+')


class CAMeLBackend(DiacritizationBackend):
    def __init__(self, db_name: str = "calima-msa-r13"):
        self._disambig = MLEDisambiguator.pretrained(db_name)

    def infer(self, sentences: list[str]) -> list[str]:
        results = []
        for sentence in sentences:
            tokens = _WHITESPACE_RE.findall(sentence)
            out_tokens = []
            for tok in tokens:
                if not tok.strip():
                    out_tokens.append(tok)
                else:
                    subtoks = simple_word_tokenize(tok)
                    disambig = self._disambig.disambiguate(subtoks)
                    out_tokens.extend(
                        d.analyses[0].analysis.get("diac", d.word)
                        for d in disambig
                    )
            results.append("".join(out_tokens))
        return results
```

- [ ] **Step 4: Install camel_data if not already done**

```bash
camel_data -i light
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_backends_camel.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add diac/backends/camel.py tests/test_backends_camel.py
git commit -m "feat: add CAMeLBackend MLE inference adapter"
```

---

## Task 6: cli.py — Typer CLI with three subcommands

**Files:**
- Create: `diac/cli.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes: `get_backend` from `diac.backends.base`; `load_sentences`, `make_pairs` from `diac.data`; `compute_der` from `diac.metrics`
- Produces: `app` Typer object; `diac` console entry point usable end-to-end.

- [ ] **Step 1: Write CLI smoke tests**

```python
# tests/test_cli.py
import tempfile
from pathlib import Path
from typer.testing import CliRunner
from diac.cli import app

runner = CliRunner()

SAMPLE_DIAC = "فَإِنْ لَمْ يَكُونَا\n"
SAMPLE_UNDIAC = "فإن لم يكونا\n"

def test_infer_camel(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()

def test_evaluate_camel(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code == 0
    assert "DER" in result.output

def test_finetune_camel_raises_gracefully(tmp_path):
    tr = tmp_path / "train.txt"
    dv = tmp_path / "dev.txt"
    tr.write_text(SAMPLE_DIAC, encoding="utf-8")
    dv.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["finetune", "--model", "camel",
                                  "--train", str(tr), "--dev", str(dv)])
    assert result.exit_code != 0
    assert "not support" in result.output.lower() or "not support" in str(result.exception).lower()
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_cli.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement cli.py**

```python
# diac/cli.py
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional
import typer
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
):
    backend = get_backend(model)
    if checkpoint:
        backend.load(str(checkpoint))
    sentences = load_sentences(input)
    results = backend.infer(sentences)
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
):
    backend = get_backend(model)
    if checkpoint:
        backend.load(str(checkpoint))
    sentences = load_sentences(input)
    hyp = backend.infer(sentences)
    ref_lines = load_sentences(ref)
    scores = compute_der(hyp, ref_lines)
    for k, v in scores.items():
        typer.echo(f"{k}: {v:.4f}")
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: 3 tests PASS (camel must be functional from Task 5).

- [ ] **Step 5: Commit**

```bash
git add diac/cli.py tests/test_cli.py
git commit -m "feat: add Typer CLI with infer/finetune/evaluate subcommands"
```

---

## Task 7: backends/byt5.py — ByT5/Fine-Tashkeel HF adapter

**Files:**
- Create: `diac/backends/byt5.py`

**Interfaces:**
- Consumes: `DiacritizationBackend` from `diac.backends.base`
- Produces: `class ByT5Backend(DiacritizationBackend)` with:
  - `infer(sentences)` using `AutoModelForSeq2SeqLM` greedy decode
  - `finetune(train, dev, output_dir, epochs, batch_size, **kwargs)` using `Seq2SeqTrainer`
  - `save(path)` / `load(path)` via HF `save_pretrained` / `from_pretrained`

**HF checkpoint:** `"basharalrfooh/Fine-Tashkeel"` (ByT5-based seq2seq). Verify the model ID exists before running: `python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('basharalrfooh/Fine-Tashkeel')"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends_byt5.py
import pytest
from diac.backends.byt5 import ByT5Backend

@pytest.fixture(scope="module")
def backend():
    b = ByT5Backend()  # loads pretrained from HF Hub
    return b

def test_infer_returns_same_length(backend):
    results = backend.infer(["فإن لم يكونا", "قال"])
    assert len(results) == 2

def test_infer_output_contains_arabic(backend):
    result = backend.infer(["كتب"])[0]
    assert any("\u0600" <= c <= "\u06ff" for c in result)

def test_save_load_roundtrip(backend, tmp_path):
    backend.save(str(tmp_path / "byt5_ckpt"))
    b2 = ByT5Backend.__new__(ByT5Backend)
    b2.load(str(tmp_path / "byt5_ckpt"))
    result = b2.infer(["كتب"])
    assert len(result) == 1
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_backends_byt5.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement byt5.py**

```python
# diac/backends/byt5.py
from __future__ import annotations
from pathlib import Path
from diac.backends.base import DiacritizationBackend

_DEFAULT_CHECKPOINT = "basharalrfooh/Fine-Tashkeel"
_MAX_NEW_TOKENS = 512


class ByT5Backend(DiacritizationBackend):
    def __init__(self, checkpoint: str = _DEFAULT_CHECKPOINT):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(self._device)
        self._model.eval()

    def infer(self, sentences: list[str]) -> list[str]:
        import torch
        results = []
        for sent in sentences:
            inputs = self._tokenizer(sent, return_tensors="pt").to(self._device)
            with torch.no_grad():
                out = self._model.generate(**inputs, max_new_tokens=_MAX_NEW_TOKENS)
            results.append(self._tokenizer.decode(out[0], skip_special_tokens=True))
        return results

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        output_dir: str = "checkpoints/byt5",
        epochs: int = 3,
        batch_size: int = 8,
        **kwargs,
    ) -> None:
        from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
        from torch.utils.data import Dataset as TorchDataset

        class _PairDataset(TorchDataset):
            def __init__(self_, pairs, tokenizer):
                self_.pairs = pairs
                self_.tok = tokenizer

            def __len__(self_):
                return len(self_.pairs)

            def __getitem__(self_, idx):
                src, tgt = self_.pairs[idx]
                enc = self_.tok(src, truncation=True, max_length=512)
                with self_.tok.as_target_tokenizer():
                    label = self_.tok(tgt, truncation=True, max_length=512)["input_ids"]
                enc["labels"] = label
                return enc

        train_ds = _PairDataset(train, self._tokenizer)
        dev_ds   = _PairDataset(dev,   self._tokenizer)
        collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model, padding=True)

        args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            predict_with_generate=True,
            fp16=False,
            logging_steps=50,
            load_best_model_at_end=True,
        )
        trainer = Seq2SeqTrainer(
            model=self._model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=dev_ds,
            tokenizer=self._tokenizer,
            data_collator=collator,
        )
        trainer.train()
        self._model = trainer.model

    def save(self, path: str) -> None:
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)
        self._model.eval()
```

- [ ] **Step 4: Run tests** (requires internet to download checkpoint on first run)

```bash
pytest tests/test_backends_byt5.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add diac/backends/byt5.py tests/test_backends_byt5.py
git commit -m "feat: add ByT5Backend with HF Trainer fine-tuning"
```

---

## Task 8: backends/rababa.py — Rababa modern inference + fine-tuning adapter

**Files:**
- Create: `diac/backends/rababa.py`

**Interfaces:**
- Consumes: `DiacritizationBackend` from `diac.backends.base`; Rababa internals from `rababa.*` (installed via `pip install -e codes/rababa/`)
- Produces: `class RababaBackend(DiacritizationBackend)` with working `infer` and `finetune`.

**Rababa internals used:**
- `rababa.encoder.ArabicEncoder` — `encode(text) -> list[int]`, `clean(text) -> str`
- `rababa.constants.TARGET_VOCAB` — maps prediction IDs to haraqat strings
- `rababa.models.base.build_model(cfg_dict)` — constructs model from OmegaConf dict
- `rababa.config.load_task_config(task)`, `rababa.config.to_dict(cfg)`
- `rababa.training.train_supervised(train_loader, val_loader, cfg, device, ckpt_root)`
- `rababa.tasks.build_supervised_loaders(cfg, batch_size, num_workers)` — expects data in `cfg.data.root` as `train.txt`/`val.txt`

**Important:** `build_supervised_loaders` reads from disk. For fine-tuning, write your train/dev pairs to temp files under a temp data root and set `cfg.data.root` accordingly before calling it.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backends_rababa.py
import pytest
from pathlib import Path
from diac.backends.rababa import RababaBackend

SAMPLE = ["فإن لم يكونا", "قال الإسنوي"]

def test_infer_returns_correct_length():
    b = RababaBackend()  # no checkpoint → random weights, but shape correct
    results = b.infer(SAMPLE)
    assert len(results) == len(SAMPLE)

def test_infer_output_nonempty():
    b = RababaBackend()
    result = b.infer(["كتب"])[0]
    assert len(result) > 0

def test_save_load_roundtrip(tmp_path):
    b = RababaBackend()
    b.save(str(tmp_path / "rababa.pt"))
    b2 = RababaBackend()
    b2.load(str(tmp_path / "rababa.pt"))
    assert b2.infer(["كتب"])[0]  # just check it runs
```

- [ ] **Step 2: Run to verify failure**

```bash
pytest tests/test_backends_rababa.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement rababa.py**

```python
# diac/backends/rababa.py
from __future__ import annotations
import tempfile
from pathlib import Path
import torch
from diac.backends.base import DiacritizationBackend
from rababa.config import load_task_config, to_dict
from rababa.models.base import build_model
from rababa.encoder import ArabicEncoder
from rababa.constants import TARGET_VOCAB, INPUT_VOCAB

_TASK = "rababa_arabic"
_PAD_ID = 0


class RababaBackend(DiacritizationBackend):
    def __init__(self, task: str = _TASK):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_task_config(task)
        self._cfg_dict = to_dict(cfg)
        self._model = build_model(self._cfg_dict).to(self._device)
        self._model.eval()
        self._encoder = ArabicEncoder(cleaner="arabic")
        self._target_vocab = TARGET_VOCAB

    def infer(self, sentences: list[str]) -> list[str]:
        results = []
        for sent in sentences:
            clean = self._encoder.clean(sent)
            ids = self._encoder.encode(clean)
            if not ids:
                results.append(sent)
                continue
            src = torch.tensor([ids], dtype=torch.long).to(self._device)
            lengths = torch.tensor([len(ids)]).to(self._device)
            with torch.no_grad():
                outputs = self._model.forward_heads(src, lengths)
            preds = outputs[0].argmax(dim=-1)[0].tolist()
            # Reconstruct: interleave valid input chars with predicted haraqat
            valid_chars = [c for c in clean if c in set(INPUT_VOCAB)]
            out = ""
            for i, ch in enumerate(valid_chars):
                out += ch
                if i < len(preds):
                    pid = preds[i]
                    if 0 < pid < len(self._target_vocab) - 1:
                        out += self._target_vocab[pid]
            results.append(out)
        return results

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        output_dir: str = "checkpoints/rababa",
        epochs: int = 3,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        from rababa.training import train_supervised
        from rababa.tasks import build_supervised_loaders

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "train.txt").write_text(
                "\n".join(tgt for _, tgt in train), encoding="utf-8"
            )
            (data_root / "val.txt").write_text(
                "\n".join(tgt for _, tgt in dev), encoding="utf-8"
            )
            cfg = load_task_config(_TASK)
            cfg.data.root = str(data_root)
            if epochs:
                cfg.train.epochs = epochs
            cfg_dict = to_dict(cfg)
            train_loader, val_loader = build_supervised_loaders(
                cfg, batch_size=batch_size, num_workers=0
            )
            ckpt_root = Path(output_dir) / "checkpoints"
            train_supervised(
                train_loader=train_loader,
                val_loader=val_loader,
                cfg=cfg_dict,
                device=self._device,
                ckpt_root=ckpt_root,
            )
            best = ckpt_root / "best.pt"
            if best.exists():
                self.load(str(best))

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_backends_rababa.py -v
```

Expected: 3 tests PASS (with random-weight model — output is gibberish but structure is correct).

- [ ] **Step 5: Commit**

```bash
git add diac/backends/rababa.py tests/test_backends_rababa.py
git commit -m "feat: add RababaBackend with in-process infer and train_supervised"
```

---

## Task 9: backends/catt.py — CATT char-BERT inference + fine-tuning adapter

**Files:**
- Create: `diac/backends/catt.py`

**Prerequisites:** Clone and install CATT before starting this task:
```bash
git clone https://github.com/abjadai/catt codes/catt
pip install -e codes/catt/
```
Then inspect `codes/catt/` to confirm the public API (trainer class, model class, tokenizer). CATT's API may differ from the stubs below — adjust imports after reading `codes/catt/` source.

**Interfaces:**
- Consumes: `DiacritizationBackend`; CATT internals from `catt.*` (post-install)
- Produces: `class CATTBackend(DiacritizationBackend)` with `infer` and `finetune`.

**CATT checkpoint:** Download `best_ed_mlm_ns_epoch_178.pt` (encoder-decoder) or `best_eo_mlm_ns_epoch_193.pt` (encoder-only) from the abjadai/catt GitHub releases. Pass the path via `CATTBackend(checkpoint_path=...)` or `backend.load(path)`.

- [ ] **Step 1: Inspect CATT source to find public API**

```bash
ls codes/catt/
python -c "import catt; print(dir(catt))"
# Read codes/catt/README.md and codes/catt/ source to identify:
# - How to load a checkpoint
# - How to run inference (diacritize a string)
# - How to run training / fine-tuning
```

Document the actual import paths before writing any code.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_backends_catt.py
import pytest
from diac.backends.catt import CATTBackend

# NOTE: set CATT_CHECKPOINT env var to a downloaded .pt file path,
# or tests will run with a randomly-initialized model (smoke test only).
import os
CKPT = os.environ.get("CATT_CHECKPOINT", None)

@pytest.fixture(scope="module")
def backend():
    b = CATTBackend()
    if CKPT:
        b.load(CKPT)
    return b

def test_infer_length(backend):
    results = backend.infer(["فإن لم يكونا", "قال"])
    assert len(results) == 2

def test_infer_nonempty(backend):
    result = backend.infer(["كتب"])[0]
    assert len(result) > 0

def test_save_load_roundtrip(backend, tmp_path):
    backend.save(str(tmp_path / "catt.pt"))
    b2 = CATTBackend()
    b2.load(str(tmp_path / "catt.pt"))
    assert b2.infer(["كتب"])[0]
```

- [ ] **Step 3: Run to verify failure**

```bash
pytest tests/test_backends_catt.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement catt.py**

The implementation depends on CATT's actual API (discovered in Step 1). Use this as the template and fill in the real CATT import paths:

```python
# diac/backends/catt.py
from __future__ import annotations
import torch
from diac.backends.base import DiacritizationBackend

# Replace these with real imports after inspecting codes/catt/ source:
# from catt.model import CATTModel
# from catt.tokenizer import CATTTokenizer
# from catt.trainer import CATTTrainer


class CATTBackend(DiacritizationBackend):
    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize model and tokenizer using CATT's API.
        # Example (adjust to actual API):
        # self._model = CATTModel().to(self._device)
        # self._tokenizer = CATTTokenizer()
        raise NotImplementedError(
            "Complete CATTBackend.__init__ after inspecting codes/catt/ source. "
            "See Task 9 Step 1."
        )

    def infer(self, sentences: list[str]) -> list[str]:
        # Use CATT's inference API to diacritize each sentence.
        raise NotImplementedError

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        output_dir: str = "checkpoints/catt",
        epochs: int = 3,
        batch_size: int = 16,
        **kwargs,
    ) -> None:
        # Use CATT's trainer to fine-tune on (undiac, diac) pairs.
        raise NotImplementedError

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
```

- [ ] **Step 5: Fill in real implementation from CATT source, then run tests**

```bash
pytest tests/test_backends_catt.py -v
```

Expected: 3 tests PASS once `__init__` is filled in.

- [ ] **Step 6: Commit**

```bash
git add diac/backends/catt.py tests/test_backends_catt.py
git commit -m "feat: add CATTBackend with abjadai/catt char-BERT"
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|---|---|
| `diac infer --model X --input f --output f --checkpoint p` | Task 6 |
| `diac finetune --model X --train f --dev f --output-dir d --epochs N --batch-size N` | Task 6 |
| `diac evaluate --model X --ref f --input f` | Task 6 |
| `camel` inference only, finetune raises clear error | Task 5, Task 6 |
| `data.load()` one-sentence-per-line | Task 2 |
| `dediac_ar` used to produce (stripped, diacritized) pairs for training | Task 2 |
| DER/WER via `diacritization-evaluation` | Task 3 |
| `DiacritizationBackend` ABC with infer/finetune/save/load | Task 4 |
| CAMeL MLE adapter | Task 5 |
| ByT5 HF adapter with Seq2SeqTrainer | Task 7 |
| Rababa modern adapter with train_supervised | Task 8 |
| CATT adapter | Task 9 |
| Apache 2.0 LICENSE | Task 1 |
| Python 3.11+, PyTorch 2.4+, no TF | Task 1 (pyproject.toml) |

All spec requirements covered.

**Placeholder check:** Task 9 has intentional stubs for CATT that require inspecting the cloned repo first. This is documented explicitly, not left as a silent TBD.

**Type consistency:** `finetune(train: list[tuple[str,str]], dev: list[tuple[str,str]], ...)` is consistent across `base.py`, `byt5.py`, `rababa.py`, `catt.py`, and `cli.py`'s `make_pairs` output.
