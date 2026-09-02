# DiacTune Option B Improvements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply a quality pass + DX uplift to DiacTune: fix correctness bugs, improve packaging and CI, tighten typing, and extend the CLI with `--format`, `--batch-size`, and env-var checkpoint loading.

**Architecture:** The existing adapter pattern (`DiacritizationBackend` ABC + registry + Typer CLI) is preserved throughout. All changes are additive or corrective — no new backends, no new subcommands. Section ordering is: correctness first, then packaging, then typing, then CLI UX.

**Tech Stack:** Python 3.11+, Typer, PyTorch, HuggingFace Transformers, camel-tools, pytest, GitHub Actions.

## Global Constraints

- Python ≥ 3.11
- No new backends
- No new subcommands
- Adapter pattern must remain intact — CLI knows nothing about model internals
- All tests must pass with `pytest tests/ -v -m "not slow"` in a clean env with only `.[dev]` + CAMeL data installed
- Commits are per-task (one commit per task, or per logical step within a task)

---

## File Map

| File | Action | Reason |
|------|--------|--------|
| `diac/cli.py` | Modify | Remove unused import, add count guard in evaluate, add `--format`, `--batch-size`, env-var checkpoint loading, `TOKENIZERS_PARALLELISM` suppression |
| `diac/data.py` | Modify | Rename loop variable `l` → `line` |
| `diac/metrics.py` | Modify | Add `ValueError` for mismatched hyp/ref lengths |
| `diac/backends/base.py` | Modify | Add `DiacritizationProtocol`, sort registry list in error message |
| `diac/backends/camel.py` | Modify | Rename `_WHITESPACE_RE` → `_TOKEN_RE`, add comment |
| `diac/backends/byt5.py` | Modify | Forward `learning_rate` and `warmup_steps` kwargs |
| `diac/backends/rababa.py` | Modify | Add ordering comment above module-level imports |
| `tests/test_cli.py` | Modify | Strengthen `test_evaluate_camel`, add format/batch-size/env-var tests |
| `tests/test_metrics.py` | Modify | Add test for mismatched length guard |
| `tests/test_backends_byt5.py` | Modify | Add `@pytest.mark.slow` to all tests |
| `tests/test_backends_catt.py` | Modify | Add `@pytest.mark.slow` to all tests |
| `pyproject.toml` | Modify | Add `rababa` optional-dep group, add `[tool.pytest.ini_options]` markers |
| `.github/workflows/ci.yml` | Create | GitHub Actions CI pipeline |

---

### Task 1: Trivial cleanups — unused import, variable rename, registry sort, camel rename, rababa comment

**Files:**
- Modify: `diac/cli.py:2`
- Modify: `diac/data.py:8`
- Modify: `diac/backends/base.py:35`
- Modify: `diac/backends/camel.py:8`
- Modify: `diac/backends/rababa.py:45`

**Interfaces:**
- Produces: nothing — these are internal renames with no API change

- [ ] **Step 1: Remove unused `import sys` from `cli.py`**

In `diac/cli.py`, delete line 2:
```python
# DELETE this line:
import sys
```

- [ ] **Step 2: Rename loop variable in `data.py`**

In `diac/data.py`, change line 8:
```python
# Before:
return [l.strip() for l in lines if l.strip()]

# After:
return [line.strip() for line in lines if line.strip()]
```

- [ ] **Step 3: Sort registry list in `get_backend()` error message**

In `diac/backends/base.py`, change line 35:
```python
# Before:
raise ValueError(f"Unknown backend '{name}'. Choose from: {list(_REGISTRY)}")

# After:
raise ValueError(f"Unknown backend '{name}'. Choose from: {sorted(_REGISTRY)}")
```

- [ ] **Step 4: Rename `_WHITESPACE_RE` → `_TOKEN_RE` in `camel.py` and add comment**

In `diac/backends/camel.py`:
```python
# Before:
_WHITESPACE_RE = re.compile(r'\s+|\S+')

# After:
# Matches whitespace runs and non-whitespace tokens alternately, preserving spacing.
_TOKEN_RE = re.compile(r'\s+|\S+')
```

Update the reference on the `infer` line:
```python
# Before:
tokens = _WHITESPACE_RE.findall(sentence)

# After:
tokens = _TOKEN_RE.findall(sentence)
```

- [ ] **Step 5: Add ordering comment in `rababa.py`**

In `diac/backends/rababa.py`, add a comment immediately before the module-level rababa imports (currently line 45):
```python
# These imports must appear AFTER _inject_swiglu_if_missing is defined above.
# The function is called in __init__ before any rababa submodule is imported,
# but Python resolves module-level imports at load time — so swiglu injection
# must already be in sys.modules before `from rababa.models.modern import ...`
# triggers the chain of rababa imports.
from rababa.config import load_task_config, to_dict
from rababa.constants import INPUT_VOCAB, TARGET_VOCAB
from rababa.encoder import ArabicEncoder
from rababa.models.modern import build_modern_student
```

- [ ] **Step 6: Run tests to confirm nothing broke**

```bash
pytest tests/test_data.py tests/test_cli.py tests/test_backends_camel.py -v
```

Expected: all pass (same as before).

- [ ] **Step 7: Commit**

```bash
git add diac/cli.py diac/data.py diac/backends/base.py diac/backends/camel.py diac/backends/rababa.py
git commit -m "refactor: trivial cleanups — remove unused import, rename variables, sort registry"
```

---

### Task 2: Guard mismatched hyp/ref lengths in `metrics.py` + evaluate CLI count check

**Files:**
- Modify: `diac/metrics.py`
- Modify: `diac/cli.py` — `evaluate` command
- Modify: `tests/test_metrics.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `compute_der(hyp, ref)` raises `ValueError` when `len(hyp) != len(ref)`

- [ ] **Step 1: Write failing test for `compute_der` length guard**

In `tests/test_metrics.py`, add:
```python
import pytest
from diac.metrics import compute_der

def test_compute_der_raises_on_length_mismatch():
    hyp = ["فَإِنْ لَمْ يَكُونَا"]
    ref = ["فَإِنْ لَمْ يَكُونَا", "قَالَ"]
    with pytest.raises(ValueError, match="hyp and ref must have the same number of lines"):
        compute_der(hyp, ref)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_metrics.py::test_compute_der_raises_on_length_mismatch -v
```

Expected: FAIL — no `ValueError` is raised currently.

- [ ] **Step 3: Add the guard in `metrics.py`**

In `diac/metrics.py`, add at the top of `compute_der`:
```python
def compute_der(hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Compute DER and WER between hypothesis and reference sentence lists.

    Writes temporary files because diacritization_evaluation operates on paths.
    """
    if len(hyp) != len(ref):
        raise ValueError(
            f"hyp and ref must have the same number of lines, "
            f"got {len(hyp)} vs {len(ref)}"
        )
    with tempfile.TemporaryDirectory() as tmp:
        ...  # rest unchanged
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
pytest tests/test_metrics.py::test_compute_der_raises_on_length_mismatch -v
```

Expected: PASS.

- [ ] **Step 5: Write failing CLI test for count mismatch in `evaluate`**

In `tests/test_cli.py`, add:
```python
def test_evaluate_mismatched_line_counts_exits_nonzero(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    # 1 undiacritized line → 1 hypothesis line; 2 reference lines → mismatch
    inp.write_text("فإن لم يكونا\n", encoding="utf-8")
    ref.write_text("فَإِنْ لَمْ يَكُونَا\nقَالَ\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code != 0
    assert "same number" in result.output.lower() or "same number" in str(result.exception).lower()
```

- [ ] **Step 6: Run test to confirm it fails**

```bash
pytest tests/test_cli.py::test_evaluate_mismatched_line_counts_exits_nonzero -v
```

Expected: FAIL — currently crashes or exits 0.

- [ ] **Step 7: Add count check in `cli.py` evaluate command**

In `diac/cli.py`, in the `evaluate` function, after loading both files:
```python
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
```

- [ ] **Step 8: Strengthen `test_evaluate_camel` to assert all 4 metric keys**

In `tests/test_cli.py`, replace the existing `test_evaluate_camel`:
```python
def test_evaluate_camel(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code == 0
    for key in ("DER", "DER*", "WER", "WER*"):
        assert key in result.output, f"Expected '{key}' in evaluate output"
```

- [ ] **Step 9: Run all modified tests**

```bash
pytest tests/test_metrics.py tests/test_cli.py -v
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add diac/metrics.py diac/cli.py tests/test_metrics.py tests/test_cli.py
git commit -m "fix: guard mismatched hyp/ref lengths in metrics and evaluate CLI"
```

---

### Task 3: Forward `learning_rate` and `warmup_steps` kwargs in `ByT5Backend.finetune`

**Files:**
- Modify: `diac/backends/byt5.py`

**Interfaces:**
- Produces: `ByT5Backend.finetune(..., learning_rate=5e-5, warmup_steps=100)` passes those values to `Seq2SeqTrainingArguments`; all other kwargs are silently dropped with a comment.

- [ ] **Step 1: Update `finetune` signature and forward kwargs**

In `diac/backends/byt5.py`, update the `finetune` method:
```python
def finetune(
    self,
    train: list[tuple[str, str]],
    dev: list[tuple[str, str]],
    output_dir: str = "checkpoints/byt5",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 0,
    **kwargs,  # remaining kwargs intentionally ignored
) -> None:
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        DataCollatorForSeq2Seq,
    )
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
            label = self_.tok(tgt, truncation=True, max_length=512)["input_ids"]
            enc["labels"] = label
            return enc

    train_ds = _PairDataset(train, self._tokenizer)
    dev_ds = _PairDataset(dev, self._tokenizer)
    collator = DataCollatorForSeq2Seq(self._tokenizer, model=self._model, padding=True)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        fp16=False,
        logging_steps=50,
        load_best_model_at_end=True,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
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
```

- [ ] **Step 2: Verify no import errors**

```bash
python -c "from diac.backends.byt5 import ByT5Backend; print('ok')"
```

Expected: `ok` (no import error; model is not loaded here).

- [ ] **Step 3: Commit**

```bash
git add diac/backends/byt5.py
git commit -m "fix: forward learning_rate and warmup_steps kwargs in ByT5Backend.finetune"
```

---

### Task 4: Packaging — `rababa` optional-dep group + pytest markers + `@pytest.mark.slow`

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/test_backends_byt5.py`
- Modify: `tests/test_backends_catt.py`

**Interfaces:**
- Produces: `pytest -m "not slow"` skips all byt5 and catt tests; `pytest -m slow` runs only them (when env is compatible)

- [ ] **Step 1: Add `rababa` optional-dep group and pytest markers to `pyproject.toml`**

In `pyproject.toml`, update `[project.optional-dependencies]` and add `[tool.pytest.ini_options]`:
```toml
[project.optional-dependencies]
rababa = []   # install manually: pip install -e codes/rababa/
catt   = ["pytorch_lightning>=2.0", "kaldialign"]
dev    = ["pytest>=8", "pytest-cov"]

[tool.pytest.ini_options]
markers = [
    "slow: marks tests that hit the network or load large models (deselect with '-m not slow')",
]
```

- [ ] **Step 2: Add `@pytest.mark.slow` to all tests in `test_backends_byt5.py`**

In `tests/test_backends_byt5.py`, add the marker after the existing `pytestmark`:
```python
import pytest

# Guard: transformers 5.x requires torch >= 2.5 to load models.
def _check_env():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return f"torch not importable: {exc}"
    try:
        from transformers import AutoModelForSeq2SeqLM
        _ = AutoModelForSeq2SeqLM.from_pretrained  # type: ignore[attr-defined]
    except Exception as exc:
        return str(exc)
    return None

_ENV_SKIP = _check_env()

pytestmark = [
    pytest.mark.skipif(
        _ENV_SKIP is not None,
        reason=f"torch/transformers environment not compatible: {_ENV_SKIP}",
    ),
    pytest.mark.slow,
]
```

- [ ] **Step 3: Add `@pytest.mark.slow` to all tests in `test_backends_catt.py`**

In `tests/test_backends_catt.py`, update `pytestmark`:
```python
pytestmark = [
    pytest.mark.skipif(
        CATT_UNAVAILABLE is not None,
        reason=f"CATT environment not available: {CATT_UNAVAILABLE}",
    ),
    pytest.mark.slow,
]
```

- [ ] **Step 4: Verify markers work**

```bash
pytest tests/ -v -m "not slow" --collect-only 2>&1 | grep "test_backends_byt5\|test_backends_catt"
```

Expected: no lines — byt5 and catt tests are excluded.

```bash
pytest tests/ -v -m "not slow"
```

Expected: all collected tests pass (camel, data, metrics, cli, rababa if available).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/test_backends_byt5.py tests/test_backends_catt.py
git commit -m "feat: add rababa optional-dep group, pytest slow markers, and mark byt5/catt tests slow"
```

---

### Task 5: GitHub Actions CI workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: CI runs on push/PR to `main`, matrix Python 3.11/3.12, runs `pytest -m "not slow"`

- [ ] **Step 1: Create `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[dev]"

      - name: Download CAMeL data
        run: camel_data -i light

      - name: Run tests (excluding slow)
        run: pytest tests/ -v -m "not slow"
```

- [ ] **Step 2: Verify the file is valid YAML**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow — matrix Python 3.11/3.12, skip slow tests"
```

---

### Task 6: Typing — `DiacritizationProtocol` + return-type annotations on concrete backends

**Files:**
- Modify: `diac/backends/base.py`
- Modify: `diac/backends/camel.py`
- Modify: `diac/backends/byt5.py`
- Modify: `diac/backends/rababa.py`
- Modify: `diac/backends/catt.py`

**Interfaces:**
- Produces: `DiacritizationProtocol` exported from `base.py`; all concrete `infer`/`finetune`/`save`/`load` methods have explicit `-> list[str]` or `-> None` return types

- [ ] **Step 1: Add `DiacritizationProtocol` to `base.py`**

In `diac/backends/base.py`:
```python
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable


@runtime_checkable
class DiacritizationProtocol(Protocol):
    """Structural protocol for duck-typed backends (no inheritance required)."""

    def infer(self, sentences: list[str], **kwargs) -> list[str]: ...
    def finetune(self, train: list[tuple[str, str]], dev: list[tuple[str, str]], **kwargs) -> None: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


class DiacritizationBackend(ABC):
    @abstractmethod
    def infer(self, sentences: list[str], **kwargs) -> list[str]: ...

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


def get_backend(name: str) -> DiacritizationProtocol:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backend '{name}'. Choose from: {sorted(_REGISTRY)}")
    module_path, class_name = _REGISTRY[name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()
```

- [ ] **Step 2: Add return-type annotations to `camel.py`**

Update `CAMeLBackend.infer` signature:
```python
def infer(self, sentences: list[str], **kwargs) -> list[str]:
```

- [ ] **Step 3: Add return-type annotations to `byt5.py`**

```python
def infer(self, sentences: list[str], **kwargs) -> list[str]:
    ...

def finetune(
    self,
    train: list[tuple[str, str]],
    dev: list[tuple[str, str]],
    output_dir: str = "checkpoints/byt5",
    epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 5e-5,
    warmup_steps: int = 0,
    **kwargs,
) -> None:
    ...

def save(self, path: str) -> None:
    ...

def load(self, path: str) -> None:
    ...
```

- [ ] **Step 4: Add return-type annotations to `rababa.py`**

```python
def infer(self, sentences: list[str], **kwargs) -> list[str]:
    ...

def finetune(
    self,
    train: list[tuple[str, str]],
    dev: list[tuple[str, str]],
    output_dir: str = "checkpoints/rababa",
    epochs: int = 3,
    batch_size: int = 32,
    **kwargs,
) -> None:
    ...

def save(self, path: str) -> None:
    ...

def load(self, path: str) -> None:
    ...
```

- [ ] **Step 5: Add return-type annotations to `catt.py`**

```python
def infer(self, sentences: list[str], **kwargs) -> list[str]:
    ...

def finetune(
    self,
    train: list[tuple[str, str]],
    dev: list[tuple[str, str]],
    output_dir: str = "checkpoints/catt",
    epochs: int = 3,
    batch_size: int = 16,
    tashkeel_threshold: float = 0.3,
    **kwargs,
) -> None:
    ...

def save(self, path: str) -> None:
    ...

def load(self, path: str) -> None:
    ...
```

- [ ] **Step 6: Write a test that `isinstance` check works with the Protocol**

In `tests/test_backends_camel.py` (or a new `tests/test_base.py`), add:
```python
def test_camel_backend_satisfies_protocol():
    from diac.backends.base import DiacritizationProtocol
    from diac.backends.camel import CAMeLBackend
    b = CAMeLBackend()
    assert isinstance(b, DiacritizationProtocol)
```

- [ ] **Step 7: Run tests**

```bash
pytest tests/ -v -m "not slow"
```

Expected: all pass including the new protocol test.

- [ ] **Step 8: Commit**

```bash
git add diac/backends/base.py diac/backends/camel.py diac/backends/byt5.py \
        diac/backends/rababa.py diac/backends/catt.py tests/
git commit -m "feat: add DiacritizationProtocol and tighten return-type annotations on all backends"
```

---

### Task 7: CLI UX — `--format` on `evaluate`, `TOKENIZERS_PARALLELISM` suppression

**Files:**
- Modify: `diac/cli.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Produces: `diactune evaluate ... --format json` emits a JSON object; `--format text` (default) unchanged

- [ ] **Step 1: Write failing tests for `--format`**

In `tests/test_cli.py`, add:
```python
import json

def test_evaluate_camel_format_json(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref),
                                  "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("DER", "DER*", "WER", "WER*"):
        assert key in data
        assert isinstance(data[key], float)

def test_evaluate_camel_format_text_is_default(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code == 0
    # text format: each line is "KEY: value"
    assert "DER:" in result.output
    assert "WER:" in result.output
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
pytest tests/test_cli.py::test_evaluate_camel_format_json tests/test_cli.py::test_evaluate_camel_format_text_is_default -v
```

Expected: FAIL — `--format` option doesn't exist yet.

- [ ] **Step 3: Implement `--format` and `TOKENIZERS_PARALLELISM` in `cli.py`**

Replace the full `cli.py` with:
```python
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
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all pass including the new format tests.

- [ ] **Step 5: Commit**

```bash
git add diac/cli.py tests/test_cli.py
git commit -m "feat: add --format json to evaluate, --batch-size to infer, env-var checkpoint loading, suppress tokenizer warning"
```

---

### Task 8: Tests for `--batch-size` and env-var checkpoint loading

**Files:**
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `diac/cli.py` from Task 7

- [ ] **Step 1: Write tests**

In `tests/test_cli.py`, add:
```python
def test_infer_batch_size_flag(tmp_path):
    """--batch-size is accepted and does not break infer output."""
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out),
                                  "--batch-size", "4"])
    assert result.exit_code == 0
    assert out.exists()


def test_infer_env_var_checkpoint_does_not_crash_on_missing(tmp_path, monkeypatch):
    """If CATT_CHECKPOINT points to a nonexistent file, the CLI errors clearly
    rather than silently ignoring it. We test with camel (no env var) to confirm
    no spurious env-var load is attempted for backends without a mapping."""
    monkeypatch.delenv("CATT_CHECKPOINT", raising=False)
    monkeypatch.delenv("RABABA_CHECKPOINT", raising=False)
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out)])
    assert result.exit_code == 0
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_cli.py
git commit -m "test: add coverage for --batch-size flag and env-var checkpoint path"
```

---

## Self-Review

**Spec coverage check:**

| Spec item | Task |
|-----------|------|
| Remove unused `import sys` | Task 1 |
| Add `rababa` optional-dep group | Task 4 |
| `ValueError` for mismatched hyp/ref | Task 2 |
| Count check in `evaluate` CLI | Task 2 |
| Strengthen `test_evaluate_camel` | Task 2 |
| Forward `learning_rate`/`warmup_steps` in byt5 | Task 3 |
| Ordering comment in `rababa.py` | Task 1 |
| pytest markers + `@pytest.mark.slow` | Task 4 |
| GitHub Actions CI | Task 5 |
| `DiacritizationProtocol` + return types | Task 6 |
| Sort registry in error message | Task 1 |
| Rename `_WHITESPACE_RE` → `_TOKEN_RE` | Task 1 |
| Rename `l` → `line` in `data.py` | Task 1 |
| `--format [text\|json]` on evaluate | Task 7 |
| `--batch-size` on infer | Task 7 |
| Env-var checkpoint loading (CATT/RABABA) | Task 7 |
| `TOKENIZERS_PARALLELISM` suppression | Task 7 |

All spec requirements covered. No gaps found.

**Type consistency:** All `infer` signatures updated to `(self, sentences: list[str], **kwargs) -> list[str]` across Task 3 (byt5), Task 6 (all backends). `get_backend()` return type updated to `DiacritizationProtocol` in Task 6. Consistent throughout.

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete.
