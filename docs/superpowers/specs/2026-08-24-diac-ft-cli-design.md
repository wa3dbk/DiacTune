# diac-ft CLI — Design Spec

**Date:** 2026-08-24  
**Status:** Approved

---

## Overview

A unified Python CLI for inference, fine-tuning, and evaluation of Arabic diacritization systems. Targets a single researcher persona running experiments locally. Supports MSA and dialectal Arabic data (small: <10k sentences, medium: 10k–100k).

---

## Architecture

### Adapter pattern

A `DiacritizationBackend` abstract base class defines a common interface. Each toolkit/model is a concrete adapter. The CLI dispatches to adapters; adapters encapsulate all toolkit-specific logic.

```
diac-ft/
├── diac/
│   ├── cli.py            # Typer entry point — 3 subcommands
│   ├── data.py           # one-sentence-per-line loader; strips diacritics via dediac_ar
│   ├── metrics.py        # DER/WER via diacritization-evaluation package
│   └── backends/
│       ├── base.py       # DiacritizationBackend ABC
│       ├── camel.py      # CAMeL MLE adapter (infer only)
│       ├── rababa.py     # Rababa modern adapter (infer + finetune)
│       ├── catt.py       # CATT char-BERT HF adapter (infer + finetune)
│       └── byt5.py       # ByT5/Fine-Tashkeel HF adapter (infer + finetune)
├── codes/                # cloned toolkits (unchanged)
├── pyproject.toml
└── LICENSE               # Apache 2.0
```

### Backend interface

```python
class DiacritizationBackend(ABC):
    @abstractmethod
    def infer(self, sentences: list[str]) -> list[str]: ...

    def finetune(self, train: list[str], dev: list[str], **kwargs) -> None:
        raise NotImplementedError(f"{self.__class__.__name__} does not support fine-tuning")

    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

---

## CLI Surface

```bash
diac infer    --model {camel,rababa,catt,byt5} \
              --input file.txt \
              [--output out.txt] \
              [--checkpoint path/to/checkpoint]

diac finetune --model {rababa,catt,byt5} \
              --train train.txt \
              --dev dev.txt \
              [--output-dir dir/] \
              [--epochs N] \
              [--batch-size N]

diac evaluate --model {camel,rababa,catt,byt5} \
              --ref ref_diacritized.txt \
              --input undiacritized.txt
              # runs inference then computes DER/WER against ref
```

`--model camel` is inference-only. Fine-tuning attempts on `camel` raise a clear error.

---

## Data Flow

**infer:**
```
file.txt → data.load() → backend.infer(sentences) → stdout or --output file
```

**finetune:**
```
train.txt + dev.txt → data.load()
  → dediac_ar(line) → (input, target) pairs
  → backend.finetune(train_pairs, dev_pairs, **kwargs)
  → checkpoint saved to --output-dir
```

**evaluate:**
```
--input file.txt → backend.infer() → hyp lines
--ref file.txt   → ref lines
metrics.der(hyp, ref) → prints DER and WER
```

`data.load()` reads a fully diacritized `.txt` (one sentence per line). For training it uses `camel_tools.utils.dediac.dediac_ar` to strip diacritics and produce `(stripped_input, diacritized_target)` pairs. CAMeL Tools is used **only** for this preprocessing — not for evaluation.

---

## Backend Inventory

| Backend | Source | Infer | Finetune | Notes |
|---|---|---|---|---|
| `camel` | pip (`camel-tools>=1.6`) | ✓ | ✗ | MLE morphology, strong MSA baseline |
| `rababa` | `codes/rababa/src/` (local) | ✓ | ✓ | CBHG/transformer, PyTorch-native, has own train CLI |
| `catt` | `CAMeL-Lab/camel-bert` on HF Hub | ✓ | ✓ | SOTA DER on WikiNews/CATT benchmarks |
| `byt5` | `basharalrfooh/Fine-Tashkeel` on HF Hub | ✓ | ✓ | **Recommended first fine-tune target for dialectal data** — byte-level, no tokenizer OOV issues |

### Why byt5 first for dialectal data
ByT5 operates at the byte level — no tokenizer vocabulary to mismatch with dialectal or OOV Arabic words. Achieves 40% WER reduction with minimal training data. Ideal for small/medium datasets.

### Excluded toolkits
- **Shakkala**: dropped — TensorFlow 2.9 dependency conflicts; no fine-tuning API.
- **Farasapy**: excluded from public release — underlying QCRI Java jars are research-only (non-commercial). Can be added locally behind an explicit opt-in if needed.

---

## Evaluation

DER and WER are computed using the [`diacritization-evaluation`](https://pypi.org/project/diacritization-evaluation/) package (already a Rababa dependency). This is the standard in Arabic diacritization research.

`metrics.py` wraps it in a single function:
```python
def compute_der(hyp: list[str], ref: list[str]) -> dict:
    # returns {"DER": float, "WER": float}
```

---

## Environment

Single Python 3.11+ virtual environment. No TensorFlow.

```toml
[project]
requires-python = ">=3.11"

[project.dependencies]
torch = ">=2.4"
transformers = ">=4.46"
datasets = ">=3.0"
camel-tools = ">=1.6"
diacritization-evaluation = ">=0.5"
typer = ">=0.12"
omegaconf = ">=2.3"
```

Rababa is installed as a local editable package from `codes/rababa/`. CAMeL Tools is pip-installed (requires Rust compiler + CMake + Boost on macOS: `brew install cmake boost`).

---

## License

**Apache 2.0** — compatible with all retained dependencies (MIT, BSD-2, Apache). Requires attribution, suitable for public GitHub release and academic citation. Avoids GPL contamination.

---

## What's Not In Scope

- Multi-GPU / distributed training (Rababa's existing Modal-based cloud training handles this if needed)
- Serving / REST API
- Interactive REPL or web UI
- Support for languages other than Arabic
