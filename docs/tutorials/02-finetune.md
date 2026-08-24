# Tutorial 2: Fine-Tuning on Dialectal Data

Adapt a pre-trained diacritization model to your domain or dialect using
DiacTune's `finetune` command.

## Supported Backends

| Backend | Fine-tune |
|---------|-----------|
| `camel` | No (morphology-based, no gradient training) |
| `byt5`  | Yes |
| `rababa`| Yes |
| `catt`  | Yes (requires checkpoint + separate setup) |

## Data Format

Your training and validation files must be **fully diacritized**, one sentence
per line (UTF-8). DiacTune automatically strips the diacritics to create
(undiacritized input, diacritized target) training pairs — you do not need to
prepare two files manually.

```
# data/train.txt  — diacritized, one sentence per line
هَذَا مِثَالٌ عَلَى الْبَيَانَاتِ الْمُشَكَّلَةِ
الْجُمَلُ الْقَصِيرَةُ أَسْهَلُ فِي التَّدْرِيبِ
```

Recommended sizes:

| Scale | Sentences | Use case |
|-------|-----------|----------|
| Small | < 10 k | Domain adaptation, dialect with limited data |
| Medium | 10 k – 100 k | General dialectal fine-tuning |

## Fine-Tune ByT5

```bash
diactune finetune \
  --model byt5 \
  --train data/dialect_train.txt \
  --dev   data/dialect_dev.txt \
  --output-dir checkpoints/dialect-byt5 \
  --epochs 5 \
  --batch-size 8
```

The fine-tuned model is saved in HuggingFace `save_pretrained` format under
`checkpoints/dialect-byt5/`. You can load it later with `--checkpoint`.

**Memory tip:** ByT5-small fits in 8 GB GPU RAM at batch-size 8 with sequences
up to 512 bytes.

## Fine-Tune Rababa

```bash
diactune finetune \
  --model rababa \
  --train data/dialect_train.txt \
  --dev   data/dialect_dev.txt \
  --output-dir checkpoints/dialect-rababa \
  --epochs 10 \
  --batch-size 32
```

Rababa's training loop writes checkpoints to
`checkpoints/dialect-rababa/checkpoints/best.pt` and loads the best checkpoint
automatically into the backend instance.

## Fine-Tune CATT

CATT requires a base checkpoint downloaded from
[abjadai/catt releases](https://github.com/abjadai/catt/releases).

```bash
# Install optional CATT dependencies first
pip install ".[catt]"

export CATT_CHECKPOINT=path/to/catt_base.pt

diactune finetune \
  --model catt \
  --train data/dialect_train.txt \
  --dev   data/dialect_dev.txt \
  --output-dir checkpoints/dialect-catt \
  --epochs 3 \
  --batch-size 16
```

## Use the Fine-Tuned Model

```bash
diactune infer \
  --model byt5 \
  --checkpoint checkpoints/dialect-byt5 \
  --input test_undiac.txt \
  --output predictions.txt
```

## Evaluate After Fine-Tuning

See [Tutorial 3: Evaluation](03-evaluate.md) for how to measure DER/WER.

## Tips

- Start with ByT5 for small dialectal datasets — its byte-level encoding
  handles dialectal spelling variants well with minimal data.
- Increase `--epochs` if validation loss is still falling; Rababa benefits from
  more epochs on small data.
- Always hold out a dedicated test set (separate from `--dev`) for final
  evaluation so you do not tune on the test distribution.
