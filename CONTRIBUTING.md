# Contributing to DiacTune

## Setup

```bash
git clone https://github.com/wa3dbk/DiacTune
cd DiacTune
pip install -e ".[dev]"
pip install diacritization-evaluation
camel_data -i light
```

## Running Tests

```bash
pytest tests/ -v
```

ByT5 tests are skipped unless `torch >= 2.5` is available. Rababa tests require the Rababa submodule under `codes/rababa/`.

## Adding a Backend

1. Create `diac/backends/<name>.py` with a class that inherits from `DiacritizationBackend`.
2. Implement `infer(sentences)` and optionally `finetune(...)`, `save()`, `load()`.
3. Register it in `diac/backends/base.py` under `_REGISTRY`.
4. Add tests in `tests/test_<name>.py`.

## Code Style

No enforced linter yet — keep imports sorted, functions documented at the signature level.

## Submitting Changes

Open a pull request against `main`. Include a short description of what changed and why.

## License

All contributions are licensed under Apache 2.0.
