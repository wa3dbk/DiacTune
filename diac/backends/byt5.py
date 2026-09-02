from __future__ import annotations

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

    def save(self, path: str) -> None:
        self._model.save_pretrained(path)
        self._tokenizer.save_pretrained(path)

    def load(self, path: str) -> None:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(path).to(self._device)
        self._model.eval()
