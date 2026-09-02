# DiacTune Improvements — Design Spec

**Date:** 2026-09-02  
**Status:** Approved  
**Scope:** Option B — quality pass + DX uplift

---

## Overview

A structured improvement pass over the existing DiacTune codebase. Covers four areas: bug fixes and correctness gaps (Option A foundation), packaging and CI, typing and code quality, and CLI UX. No new backends are added. The adapter pattern and CLI surface remain intact.

---

## Section 1 — Bug Fixes & Correctness

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `cli.py:2` | `import sys` is unused | Remove it |
| 2 | `pyproject.toml` | `rababa` has no optional-dep group — users get a silent import error | Add `[project.optional-dependencies] rababa = [...]` pointing at the local editable install |
| 3 | `metrics.py` | No guard when `len(hyp) != len(ref)` — library silently truncates or crashes | Add `ValueError` upfront |
| 4 | `cli.py` — `evaluate` | `load_sentences` strips blank lines; hyp/ref counts can diverge on files with trailing newlines | Assert counts match after loading; emit a clear error message |
| 5 | `test_cli.py:28` | `test_evaluate_camel` only checks `"DER" in output` | Assert all four keys: `DER`, `DER*`, `WER`, `WER*` |
| 6 | `byt5.py:finetune` | `**kwargs` accepted but silently ignored | Forward supported kwargs (`learning_rate`, `warmup_steps`) to `Seq2SeqTrainingArguments`, or explicitly document that they are intentionally dropped with a comment |
| 7 | `rababa.py:45` | Bare module-level rababa imports appear after `_inject_swiglu_if_missing` definition, making ordering implicit and fragile | Add an explicit comment making the ordering contract clear |

---

## Section 2 — Packaging & CI

### Optional dependency groups

Add a `rababa` group to `pyproject.toml`:

```toml
[project.optional-dependencies]
rababa = []   # installed via: pip install -e codes/rababa/
catt   = ["pytorch_lightning>=2.0", "kaldialign"]
dev    = ["pytest>=8", "pytest-cov"]
```

Document in README: `pip install -e codes/rababa/` must be run manually (no PyPI package).

### GitHub Actions

File: `.github/workflows/ci.yml`

- Trigger: push and pull_request to `main`
- Matrix: Python 3.11, 3.12
- Steps:
  1. Checkout
  2. `pip install -e ".[dev]"`
  3. `camel_data -i light`
  4. `pytest tests/ -v -m "not slow"` — heavy backends skipped via markers

### pytest markers

In `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests that hit the network or load large models (deselect with '-m not slow')",
]
```

Apply `@pytest.mark.slow` to:
- All tests in `test_backends_byt5.py` (downloads from HF Hub)
- All tests in `test_backends_catt.py` (loads large checkpoint if `CATT_CHECKPOINT` set)

---

## Section 3 — Typing & Code Quality

- **`Protocol`-based backend type**: introduce `DiacritizationProtocol(Protocol)` in `base.py` for structural duck-typing at the `get_backend()` call site. The existing `DiacritizationBackend` ABC is kept for internal backends (provides default `NotImplementedError` impls). External backends can satisfy the Protocol without inheriting.
- **Consistent return-type annotations**: ensure `infer`, `finetune`, `save`, `load` have explicit return types on all concrete backend classes (not just the ABC).
- **`get_backend()` error message**: change `list(_REGISTRY)` to `sorted(_REGISTRY)` for deterministic output.
- **`camel.py`**: rename `_WHITESPACE_RE` → `_TOKEN_RE`; add a one-line comment explaining the pattern matches both whitespace runs and non-whitespace tokens to preserve spacing.
- **`data.py`**: rename loop variable `l` → `line` to avoid shadowing the builtin.

---

## Section 4 — CLI UX Improvements

### `--format` on `evaluate`

```bash
diactune evaluate --model camel --input in.txt --ref ref.txt --format json
```

- `--format text` (default): current `KEY: value\n` output
- `--format json`: emit a single JSON object to stdout — useful for scripting

Implementation: add `format: str = typer.Option("text", ...)` parameter; branch on value after `compute_der`.

### `--batch-size` on `infer`

```bash
diactune infer --model byt5 --input in.txt --batch-size 8
```

- Add `batch_size: int = typer.Option(1, ...)` to the `infer` subcommand
- Pass as `backend.infer(sentences, batch_size=batch_size)` — backends that support it use it; others accept and ignore via `**kwargs` on their `infer` signature
- No backend changes required immediately; this is a forward-compatible interface extension

### Checkpoint env vars wired into CLI

When `--checkpoint` is not passed:

- `--model catt` → check `os.environ.get("CATT_CHECKPOINT")`
- `--model rababa` → check `os.environ.get("RABABA_CHECKPOINT")`

Auto-load if set. Consistent with CATT's existing documentation but currently not implemented in `cli.py`.

### Stderr/stdout hygiene

- Add `os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")` near the top of `cli.py` to suppress the HF tokenizer parallelism warning that currently bleeds into stdout on every run.
- Verify all backend training loggers write to stderr (pytorch_lightning and transformers do by default — confirm and document).

---

## What Is Not In Scope

- New backends
- Batch inference implementation inside existing backends (only the CLI interface is extended)
- `diactune benchmark` subcommand
- `diactune download` helper for model weights
- Multi-GPU / distributed training
- REST API or web UI
