# DiacTune — Developer Guide for Claude Code

## What This Project Is

A unified CLI (`diactune`) for Arabic diacritization: inference, fine-tuning, and evaluation across four backends. The adapter pattern isolates all backend-specific logic; the CLI and metrics layer know nothing about model internals.

## Architecture at a Glance

```
diac/
  cli.py               # Typer app — 3 subcommands: infer, finetune, evaluate
  data.py              # load_sentences(), make_pairs() via dediac_ar
  metrics.py           # compute_der() → {DER, DER*, WER, WER*}
  backends/
    base.py            # DiacritizationBackend ABC + _REGISTRY + get_backend()
    camel.py           # CAMeLBackend   — MLE, inference only
    byt5.py            # ByT5Backend    — HF seq2seq, inference + finetune
    rababa.py          # RababaBackend  — CBHG transformer, inference + finetune
    catt.py            # CATTBackend    — char-BERT, inference + finetune
tests/                 # One file per backend + cli + data + metrics
codes/
  rababa/              # git clone of interscript/rababa (cloned separately)
  catt/                # git clone of abjadai/catt (cloned separately)
```

## Dev Setup

```bash
pip install -e ".[dev]"
pip install diacritization-evaluation   # not in pyproject.toml yet — install manually
camel_data -i light                     # required for camel backend and dediac_ar
```

For Rababa backend:
```bash
git clone https://github.com/interscript/rababa codes/rababa
pip install -e codes/rababa/
```

For CATT backend:
```bash
git clone https://github.com/abjadai/catt codes/catt
pip install ".[catt]"   # adds pytorch_lightning + kaldialign
```

## Running Tests

```bash
pytest tests/ -v
```

Tests for unavailable backends skip automatically — a clean run without Rababa or CATT cloned will show skips, not failures.

## Known Environment Constraints

### ByT5 tests skip on torch < 2.5
Transformers 5.x requires torch ≥ 2.5 to load models. The test file has a module-level skip guard. The backend code itself is correct; only the test environment is limited.

### Rababa: `rababa.models.swiglu` is injected synthetically
`codes/rababa/rababa/models/swiglu.py` is excluded by upstream's `.gitignore`. `RababaBackend.__init__` injects a synthetic `rababa.models.swiglu` module into `sys.modules` before any Rababa imports. Do not remove `_inject_swiglu_if_missing()`.

### Rababa: never use `build_model()` from rababa
`rababa.models.base.build_model()` unconditionally imports `rababa.models.multi_head`, which does not exist. Always use `build_modern_student(cfg_dict)` from `rababa.models.modern` directly.

### CATT is not a pip package
`codes/catt/` must be cloned. `CATTBackend` injects `codes/catt/` into `sys.path` at import time. `CATT_UNAVAILABLE` (a string or `None`) is set at module level and used by tests as a skip sentinel.

### `diacritization-evaluation` uses file paths, not strings
`der.calculate_der_from_path()` and `wer.calculate_wer_from_path()` take file paths. `compute_der()` in `metrics.py` writes temp files via `tempfile.TemporaryDirectory`. Do not change this to in-memory string passing.

### `make_pairs()` strips diacritics automatically
`make_pairs(sentences)` calls `dediac_ar(s)` to produce `(undiacritized, diacritized)` pairs. Training/dev files should be **fully diacritized**; the CLI handles stripping. Users never need to prepare two separate files.

## Adding a Backend

1. Create `diac/backends/<name>.py` with a class inheriting `DiacritizationBackend`.
2. Implement `infer(self, sentences: list[str]) -> list[str]` (required).
3. Optionally override `finetune()`, `save()`, `load()`. The base raises `NotImplementedError` for all three, which the CLI catches and converts to exit code 1.
4. Register in `_REGISTRY` in `diac/backends/base.py`:
   ```python
   "mybackend": "diac.backends.mybackend.MyBackend",
   ```
5. Add tests in `tests/test_backends_<name>.py`. Use a module-level skip guard if the backend has optional dependencies.

## Test Patterns

- Backends with optional or environment-dependent imports use a `_check_env()` function at module level that returns `None` (OK) or a skip reason string. Apply via `pytestmark = pytest.mark.skipif(...)`.
- Backend fixtures are `scope="module"` to avoid reloading heavy models per test.
- Infer tests with random weights check shape/type only, not diacritic accuracy.
- `save`/`load` round-trip tests use `tmp_path`.

## CLI Contract

- `infer`: loads backend → optionally loads checkpoint → `load_sentences` → `infer` → write/stdout.
- `finetune`: loads backend → `make_pairs(load_sentences(...))` → `finetune(train_pairs, dev_pairs, output_dir=..., epochs=..., batch_size=...)`.
- `evaluate`: loads backend → optionally loads checkpoint → `infer` → `compute_der(hyp, ref_lines)` → print 4 metrics.
- `NotImplementedError` from backend methods → `typer.Exit(code=1)` with message to stderr.

## Deferred / Known Gaps

- `diacritization-evaluation` is not listed in `pyproject.toml` dependencies (must be installed manually).
- Unused `import sys` in `cli.py` (line 2).
- `test_evaluate_camel` in `test_cli.py` could assert all 4 metric keys (currently shallow).
- ByT5 `finetune()` does not forward `**kwargs` to `Seq2SeqTrainingArguments`.
- No `pytest.mark.slow` markers on tests that hit the network (ByT5 downloads from HF Hub on first run).
