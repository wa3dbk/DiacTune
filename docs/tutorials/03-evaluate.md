# Tutorial 3: Evaluation (DER / WER)

Measure the accuracy of a diacritization model on a held-out test set using
standard Arabic diacritization metrics.

## Metrics

| Metric | Definition |
|--------|-----------|
| `DER`  | Diacritic Error Rate — fraction of incorrect diacritics, including case endings |
| `DER*` | DER ignoring case endings (last haraka of each word) |
| `WER`  | Word Error Rate — fraction of words with at least one wrong diacritic |
| `WER*` | WER ignoring case endings |

`DER*` and `WER*` are the standard reported metrics in most Arabic diacritization
papers because case-ending errors are often ambiguous in context.

## Prepare Your Files

You need two files:

- **`--input`**: undiacritized text, one sentence per line
- **`--ref`**: fully diacritized gold reference, one sentence per line

Both files must have the same number of lines.

```
# test_undiac.txt (input)
ذهب الولد إلى المدرسة

# test_ref.txt (reference)
ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ
```

If your test set is fully diacritized, DiacTune will strip the diacritics
automatically when it produces predictions — you only need to prepare the
diacritized reference file.

## Run Evaluation

### CAMeL MLE baseline

```bash
diactune evaluate \
  --model camel \
  --input  data/test_undiac.txt \
  --ref    data/test_ref.txt
```

Example output:

```
DER:  0.1823
DER*: 0.1201
WER:  0.2940
WER*: 0.1873
```

### ByT5 with a fine-tuned checkpoint

```bash
diactune evaluate \
  --model byt5 \
  --checkpoint checkpoints/dialect-byt5 \
  --input  data/test_undiac.txt \
  --ref    data/test_ref.txt
```

### Rababa

```bash
diactune evaluate \
  --model rababa \
  --checkpoint checkpoints/dialect-rababa/checkpoints/best.pt \
  --input  data/test_undiac.txt \
  --ref    data/test_ref.txt
```

### CATT

```bash
export CATT_CHECKPOINT=path/to/catt_checkpoint.pt
diactune evaluate \
  --model catt \
  --input  data/test_undiac.txt \
  --ref    data/test_ref.txt
```

## Comparing Multiple Backends

Run `evaluate` for each backend and collect results:

```bash
for model in camel byt5 rababa; do
  echo "=== $model ==="
  diactune evaluate --model "$model" --input test_undiac.txt --ref test_ref.txt
done
```

## Interpreting Results

- State-of-the-art systems on Tashkeela (MSA) achieve DER* around 2–4 %.
- On dialectal data without fine-tuning, DER* is often 20–40 % — fine-tuning
  can bring this down significantly even with a few thousand sentences.
- Compare `DER` vs `DER*` to understand whether errors concentrate on case
  endings (syntactic) or stem diacritics (lexical/phonological).

## Next Steps

- Fine-tune the best baseline — see [Tutorial 2](02-finetune.md).
- Run inference at scale — see [Tutorial 1](01-inference.md).
