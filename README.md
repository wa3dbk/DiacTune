# DiacTune

[![CI](https://github.com/wa3dbk/DiacTune/actions/workflows/ci.yml/badge.svg)](https://github.com/wa3dbk/DiacTune/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

A unified Python CLI for Arabic diacritization — inference, fine-tuning, and evaluation across multiple backends.

**[Tutorials](docs/tutorials/README.md)** · **[CLI Reference](#cli-reference)** · **[Metrics](#metrics)**

```bash
diactune infer    --model byt5   --input text.txt --output diacritized.txt
diactune finetune --model byt5   --train train.txt --dev dev.txt --output-dir checkpoints/
diactune evaluate --model camel  --input test.txt  --ref reference.txt
```

## Overview

DiacTune wraps four Arabic diacritization systems behind a single CLI, making it easy to:

- **Compare** backends on your data using standardized DER/WER metrics
- **Fine-tune** pre-trained models on dialectal or domain-specific data
- **Evaluate** MSA and dialectal Arabic with one command

| Backend | `--model` | Infer | Fine-tune | Notes |
|---|---|---|---|---|
| CAMeL Tools MLE | `camel` | ✅ | ❌ | Morphology-based, strong MSA baseline |
| ByT5 / Fine-Tashkeel | `byt5` | ✅ | ✅ | Byte-level; recommended for small dialectal data |
| Rababa (CBHG) | `rababa` | ✅ | ✅ | PyTorch-native CBHG transformer |
| CATT | `catt` | ✅ | ✅ | State-of-the-art char-BERT, requires separate setup |

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get install cmake libboost-all-dev

# macOS
brew install cmake boost

# Rust compiler (required by camel-tools)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

### Install DiacTune

```bash
git clone https://github.com/wa3dbk/DiacTune
cd DiacTune
pip install -e .
```

### Install CAMeL Tools data (required for `--model camel`)

```bash
pip install camel-tools
camel_data -i light
```

### Install Rababa backend (required for `--model rababa`)

```bash
git clone https://github.com/interscript/rababa codes/rababa
pip install -e codes/rababa/
```

### Install CATT backend (required for `--model catt`)

```bash
git clone https://github.com/abjadai/catt codes/catt
pip install pytorch_lightning kaldialign
# Download a checkpoint from https://github.com/abjadai/catt/releases
# Then: diactune infer --model catt --checkpoint path/to/checkpoint.pt ...
```

## Quick Start

### 1. Diacritize a file

```bash
# Using CAMeL MLE (no checkpoint needed, MSA only)
diactune infer --model camel --input mytext.txt --output diacritized.txt

# Using ByT5 (pre-trained on Tashkeela)
diactune infer --model byt5 --input mytext.txt --output diacritized.txt

# Using a fine-tuned checkpoint
diactune infer --model byt5 --checkpoint checkpoints/dialect-byt5 --input mytext.txt
```

### 2. Evaluate (DER / WER)

Your input file must be **undiacritized**; your reference file must be **fully diacritized** (one sentence per line).

```bash
diactune evaluate --model camel --input test_undiac.txt --ref test_ref.txt
# DER:  0.1823
# DER*: 0.1201
# WER:  0.2940
# WER*: 0.1873
```

`DER*` and `WER*` are case-ending-insensitive variants (standard in Arabic diacritization research).

### 3. Fine-tune on your data

Your training file should be **fully diacritized**, one sentence per line. DiacTune strips diacritics automatically to create (input, target) pairs.

```bash
diactune finetune \
  --model byt5 \
  --train data/dialect_train.txt \
  --dev   data/dialect_dev.txt \
  --output-dir checkpoints/dialect-byt5 \
  --epochs 5 \
  --batch-size 8
```

Then evaluate the fine-tuned model:

```bash
diactune evaluate \
  --model byt5 \
  --checkpoint checkpoints/dialect-byt5 \
  --input data/dialect_test_undiac.txt \
  --ref   data/dialect_test_ref.txt
```

## Data Format

All commands accept plain text files with **one sentence per line** (UTF-8):

```
فَإِنْ لَمْ يَكُونَا كَذَلِكَ أَتَى بِمَا يَقْتَضِيهِ الْحَالُ
قَالَ الْإِسْنَوِيُّ وَسَوَاءٌ فِيمَا قَالُوهُ
```

- `--infer --input`: undiacritized text
- `--evaluate --input`: undiacritized text; `--ref`: diacritized gold reference
- `--finetune --train` / `--dev`: diacritized text (DiacTune strips diacritics to form training pairs)

## CLI Reference

```
diactune infer
  --model     TEXT    Backend: camel, byt5, rababa, catt  [required]
  --input     FILE    Input .txt (undiacritized)           [required]
  --output    FILE    Output file (default: stdout)
  --checkpoint PATH   Path to model checkpoint

diactune finetune
  --model      TEXT   Backend: byt5, rababa, catt          [required]
  --train      FILE   Training .txt (diacritized)          [required]
  --dev        FILE   Validation .txt (diacritized)        [required]
  --output-dir PATH   Checkpoint directory  [default: checkpoints]
  --epochs     INT    Training epochs       [default: 3]
  --batch-size INT    Batch size            [default: 16]

diactune evaluate
  --model      TEXT   Backend: camel, byt5, rababa, catt  [required]
  --input      FILE   Undiacritized input .txt             [required]
  --ref        FILE   Diacritized reference .txt           [required]
  --checkpoint PATH   Path to model checkpoint
```

## Metrics

DiacTune reports four standard metrics via the [`diacritization-evaluation`](https://pypi.org/project/diacritization-evaluation/) package:

| Metric | Description |
|---|---|
| `DER` | Diacritic Error Rate (per character, including case endings) |
| `DER*` | DER ignoring case endings |
| `WER` | Word Error Rate |
| `WER*` | WER ignoring case endings |

## Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions and how to add a new backend.

## License

Apache 2.0 — see [LICENSE](LICENSE).

Dependency licenses: CAMeL Tools (MIT), Rababa (BSD-2), CATT (Apache-2.0), ByT5/Fine-Tashkeel (Apache-2.0).

> **Note:** Farasapy / QCRI Farasa is intentionally excluded — its underlying Java jars are restricted to research use only and cannot be part of a publicly distributable tool.
