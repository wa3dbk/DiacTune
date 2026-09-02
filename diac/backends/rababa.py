"""Rababa modern inference + fine-tuning adapter.

Uses the ModernCharTransformer (arch='modern') directly to avoid the broken
import in rababa.models.base.build_model, which unconditionally imports
rababa.models.multi_head — a module that does not exist in this installation.
"""

from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from diac.backends.base import DiacritizationBackend


def _inject_swiglu_if_missing() -> None:
    """Inject a synthetic rababa.models.swiglu module if it is not present.

    The file codes/rababa/src/rababa/models/swiglu.py is excluded by the
    upstream .gitignore (pattern: models/), so a clean checkout of diac-ft
    will not have it. Injecting a synthetic module here makes the backend
    self-sufficient without touching codes/rababa/.
    """
    if "rababa.models.swiglu" in sys.modules:
        return
    mod = types.ModuleType("rababa.models.swiglu")

    def swiglu(
        gate: torch.Tensor,
        up: torch.Tensor,
        clamp_max: float | None = None,
    ) -> torch.Tensor:
        if clamp_max is not None:
            gate = gate.clamp(-clamp_max, clamp_max)
            up = up.clamp(-clamp_max, clamp_max)
        return F.silu(gate) * up

    mod.swiglu = swiglu  # type: ignore[attr-defined]
    sys.modules["rababa.models.swiglu"] = mod

# These imports must appear AFTER _inject_swiglu_if_missing is defined above.
# The function is called in __init__ before any rababa submodule is imported,
# but Python resolves module-level imports at load time — so swiglu injection
# must already be in sys.modules before `from rababa.models.modern import ...`
# triggers the chain of rababa imports.
from rababa.config import load_task_config, to_dict
from rababa.constants import INPUT_VOCAB, TARGET_VOCAB
from rababa.encoder import ArabicEncoder
from rababa.models.modern import build_modern_student

_TASK = "rababa_arabic"
_PAD_ID = 0


class RababaBackend(DiacritizationBackend):
    """Rababa Arabic diacritization backend using ModernCharTransformer.

    Instantiation with no arguments builds a random-weight model suitable
    for structure checks. Pass ``checkpoint`` to load trained weights.
    """

    def __init__(self, task: str = _TASK, checkpoint: str | None = None) -> None:
        _inject_swiglu_if_missing()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg = load_task_config(task)
        self._cfg = cfg
        self._cfg_dict = to_dict(cfg)
        # Build model directly from modern factory to avoid the broken
        # build_model() which unconditionally imports missing multi_head module.
        self._model = build_modern_student(self._cfg_dict).to(self._device)
        self._model.eval()
        self._encoder = ArabicEncoder(cleaner="arabic")
        self._target_vocab = TARGET_VOCAB
        self._input_vocab_set = set(INPUT_VOCAB)
        if checkpoint is not None:
            self.load(checkpoint)

    def infer(self, sentences: list[str]) -> list[str]:
        results = []
        for sent in sentences:
            clean = self._encoder.clean(sent)
            ids = self._encoder.encode(clean)
            if not ids:
                results.append(sent)
                continue
            src = torch.tensor([ids], dtype=torch.long).to(self._device)
            lengths = torch.tensor([len(ids)]).to(self._device)
            with torch.no_grad():
                outputs = self._model.forward_heads(src, lengths)
            preds = outputs[0].argmax(dim=-1)[0].tolist()
            # Reconstruct: interleave valid input chars with predicted haraqat
            valid_chars = [c for c in clean if c in self._input_vocab_set]
            out = ""
            for i, ch in enumerate(valid_chars):
                out += ch
                if i < len(preds):
                    pid = preds[i]
                    if 0 < pid < len(self._target_vocab) - 1:
                        out += self._target_vocab[pid]
            results.append(out)
        return results

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        output_dir: str = "checkpoints/rababa",
        epochs: int = 3,
        batch_size: int = 32,
        **kwargs,
    ) -> None:
        """Fine-tune on (undiacritized, diacritized) pairs.

        Writes target (diacritized) lines to temp train.txt / val.txt,
        then runs the Rababa supervised training loop.

        Note: train_supervised internally calls build_model(cfg), which is
        patched here to use build_modern_student to avoid the missing
        rababa.models.multi_head module.
        """
        import rababa.training.supervised as _supervised
        from rababa.tasks import build_supervised_loaders
        from rababa.training import train_supervised

        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            (data_root / "train.txt").write_text(
                "\n".join(tgt for _, tgt in train), encoding="utf-8"
            )
            (data_root / "val.txt").write_text(
                "\n".join(tgt for _, tgt in dev), encoding="utf-8"
            )

            # Clone the config and point it at our temp data root.
            cfg = load_task_config(_TASK)
            cfg.data.root = str(data_root)
            if epochs:
                cfg.train.epochs = epochs

            # Patch build_model in the supervised training module so that
            # train_supervised uses build_modern_student instead of the
            # broken build_model (which imports the missing multi_head module).
            _orig_build_model = _supervised.build_model

            def _patched_build_model(cfg_dict: dict) -> torch.nn.Module:
                return build_modern_student(cfg_dict)

            _supervised.build_model = _patched_build_model
            try:
                cfg_dict = to_dict(cfg)
                cfg_dict["model"]["arch"] = "modern"
                train_loader, val_loader = build_supervised_loaders(
                    cfg, batch_size=batch_size, num_workers=0
                )
                ckpt_root = Path(output_dir) / "checkpoints"
                train_supervised(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=cfg_dict,
                    device=self._device,
                    ckpt_root=ckpt_root,
                )
            finally:
                _supervised.build_model = _orig_build_model

            best = ckpt_root / "best.pt"
            if best.exists():
                self.load(str(best))

    def save(self, path: str) -> None:
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
