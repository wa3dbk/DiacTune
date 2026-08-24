# Tutorial 1: Running Inference

Diacritize Arabic text using any of DiacTune's four backends.

## Prerequisites

Install DiacTune and the backend you want to use:

```bash
pip install -e .
```

For `--model camel` (MSA only, no checkpoint needed):

```bash
camel_data -i light
```

For `--model rababa`:

```bash
git clone https://github.com/interscript/rababa codes/rababa
pip install -e codes/rababa/
```

For `--model byt5` or `--model catt`, see the [README](../../README.md).

## Prepare Your Input

Create a plain UTF-8 text file with one sentence per line. Lines should be
**undiacritized** (no harakat):

```
# input.txt
ذهب الولد إلى المدرسة
قرأت الكتاب في المساء
```

## Run Inference

### CAMeL MLE (MSA baseline, no checkpoint required)

```bash
diactune infer --model camel --input input.txt --output output.txt
```

### ByT5 (pre-trained on Tashkeela)

```bash
diactune infer --model byt5 --input input.txt --output output.txt
```

### ByT5 with a fine-tuned checkpoint

```bash
diactune infer \
  --model byt5 \
  --checkpoint checkpoints/my-dialect-byt5 \
  --input input.txt \
  --output output.txt
```

### Rababa (CBHG transformer)

```bash
diactune infer --model rababa --input input.txt
```

### CATT (requires checkpoint)

```bash
export CATT_CHECKPOINT=path/to/catt_checkpoint.pt
diactune infer --model catt --checkpoint "$CATT_CHECKPOINT" --input input.txt
```

## Check the Output

```bash
cat output.txt
```

Expected (CAMeL MLE):

```
ذَهَبَ الْوَلَدُ إِلَى الْمَدْرَسَةِ
قَرَأْتُ الْكِتَابَ فِي الْمَسَاءِ
```

## Print to Stdout Instead of a File

Omit `--output` and results are written to stdout:

```bash
diactune infer --model camel --input input.txt
```

## Notes

- CAMeL MLE is morphology-based and works best on MSA. It does not support
  fine-tuning.
- ByT5 uses a pre-trained byte-level model from HuggingFace; it handles both
  MSA and dialectal text after fine-tuning.
- All backends expect UTF-8 encoded text. Mixed-language lines are not
  filtered — only Arabic characters are passed through the diacritization
  model.
