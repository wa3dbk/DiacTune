"""CATT (Character-based Arabic Tashkeel Transformer) inference + fine-tuning adapter.

CATT is not distributed as a pip package — it is a collection of standalone scripts
cloned from https://github.com/abjadai/catt into ``codes/catt/``.  This adapter
injects that directory into ``sys.path`` at import time so that the CATT modules
(``eo_pl``, ``ed_pl``, ``tashkeel_tokenizer``, ``tashkeel_dataset``, ``utils``) can
be imported without a formal install.

Architecture choice
-------------------
The *Encoder-Only* (EO) model is used by default:
- ``n_layers=6`` — same hyperparameters as the pretrained
  ``best_eo_mlm_ns_epoch_193.pt`` checkpoint.
- Faster than the encoder-decoder (ED) variant and preferred for inference.

To use the encoder-decoder model instead, pass ``model_type="ed"`` and
``n_layers=3``.

Checkpoint download
-------------------
Download ``best_eo_mlm_ns_epoch_193.pt`` (or the ED variant) from the
``Releases`` section of https://github.com/abjadai/catt and pass its path via::

    backend = CATTBackend()
    backend.load("/path/to/best_eo_mlm_ns_epoch_193.pt")

or set the ``CATT_CHECKPOINT`` env var before running ``diac infer``.

Fine-tuning
-----------
Fine-tuning uses ``pytorch_lightning.Trainer`` with CATT's own
``TashkeelDataset``.  The dataset expects *fully diacritized* Arabic text lines
(the (undiac, diac) pairs are consumed but only the diac side is used, matching
CATT's training regime).  The dataset internally filters sentences whose
diacritic-to-letter ratio is below ``tashkeel_to_text_ratio_threshold`` — set
it lower (e.g. 0.3) if your fine-tuning data is sparsely diacritized.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch

from diac.backends.base import DiacritizationBackend

# ---------------------------------------------------------------------------
# Path injection — CATT has no setup.py / pyproject.toml so we add it manually
# ---------------------------------------------------------------------------

_CATT_CODE_DIR = Path(__file__).parent.parent.parent / "codes" / "catt"


def _ensure_catt_on_path() -> None:
    """Insert codes/catt into sys.path so CATT modules are importable."""
    catt_str = str(_CATT_CODE_DIR)
    if catt_str not in sys.path:
        sys.path.insert(0, catt_str)


def _check_catt_available() -> str | None:
    """Return an error message if the CATT source tree is missing or broken."""
    if not _CATT_CODE_DIR.exists():
        return (
            f"CATT source not found at {_CATT_CODE_DIR}. "
            "Run: git clone https://github.com/abjadai/catt codes/catt"
        )
    _ensure_catt_on_path()
    try:
        import eo_pl  # noqa: F401
        import tashkeel_tokenizer  # noqa: F401
    except ImportError as exc:
        return (
            f"CATT modules not importable ({exc}). "
            "Ensure pytorch_lightning and kaldialign are installed: "
            "pip install pytorch_lightning kaldialign"
        )
    return None


# Check once at module level so tests can use the result for skip guards
CATT_UNAVAILABLE: str | None = _check_catt_available()

# EO model hyperparameters (match best_eo_mlm_ns_epoch_193.pt)
_EO_N_LAYERS = 6
_EO_MAX_SEQ_LEN = 1024

# ED model hyperparameters (match best_ed_mlm_ns_epoch_178.pt)
_ED_N_LAYERS = 3
_ED_MAX_SEQ_LEN = 1024


class CATTBackend(DiacritizationBackend):
    """CATT Arabic diacritization backend.

    Parameters
    ----------
    model_type:
        ``"eo"`` for Encoder-Only (default, faster), ``"ed"`` for
        Encoder-Decoder (more accurate).
    n_layers:
        Number of transformer layers.  Defaults: 6 for EO, 3 for ED.
    max_seq_len:
        Maximum input sequence length in characters (default 1024).
    checkpoint:
        Optional path to a ``.pt`` checkpoint.  Equivalent to calling
        ``backend.load(checkpoint)`` after construction.
    """

    def __init__(
        self,
        model_type: str = "eo",
        n_layers: int | None = None,
        max_seq_len: int = _EO_MAX_SEQ_LEN,
        checkpoint: str | None = None,
    ) -> None:
        if CATT_UNAVAILABLE:
            raise ImportError(CATT_UNAVAILABLE)

        _ensure_catt_on_path()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model_type = model_type.lower()

        if self._model_type not in ("eo", "ed"):
            raise ValueError(f"model_type must be 'eo' or 'ed', got '{model_type}'")

        # Resolve default n_layers per model type
        if n_layers is None:
            n_layers = _EO_N_LAYERS if self._model_type == "eo" else _ED_N_LAYERS

        self._n_layers = n_layers
        self._max_seq_len = max_seq_len

        # Import CATT modules (path already injected above)
        from tashkeel_tokenizer import TashkeelTokenizer

        self._tokenizer = TashkeelTokenizer()
        self._model = self._build_model()
        self._model.eval().to(self._device)

        if checkpoint is not None:
            self.load(checkpoint)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> "pl.LightningModule":  # type: ignore[name-defined]
        if self._model_type == "eo":
            from eo_pl import TashkeelModel
        else:
            from ed_pl import TashkeelModel

        return TashkeelModel(
            self._tokenizer,
            max_seq_len=self._max_seq_len,
            n_layers=self._n_layers,
            learnable_pos_emb=False,
        )

    # ------------------------------------------------------------------
    # DiacritizationBackend interface
    # ------------------------------------------------------------------

    def infer(self, sentences: list[str]) -> list[str]:
        """Diacritize a batch of Arabic sentences.

        Empty or non-Arabic strings are returned as-is after stripping
        non-Arabic characters via CATT's ``remove_non_arabic`` utility.
        """
        from utils import remove_non_arabic

        cleaned = [remove_non_arabic(s) for s in sentences]
        results = self._model.do_tashkeel_batch(cleaned, batch_size=16, verbose=False)
        return results

    def finetune(
        self,
        train: list[tuple[str, str]],
        dev: list[tuple[str, str]],
        output_dir: str = "checkpoints/catt",
        epochs: int = 3,
        batch_size: int = 16,
        tashkeel_threshold: float = 0.3,
        **kwargs,
    ) -> None:
        """Fine-tune CATT on (undiacritized, diacritized) pairs.

        Only the diacritized side of each pair is used, matching CATT's
        training regime where the dataset internally strips and re-applies
        diacritics.  Sentences with a diacritic-to-letter ratio below
        ``tashkeel_threshold`` are filtered out by ``TashkeelDataset``.

        Checkpoints are saved under ``output_dir/`` via PyTorch Lightning's
        ``ModelCheckpoint`` callback.  After training the best checkpoint
        (by ``val_der``) is loaded back into ``self._model``.

        Parameters
        ----------
        train, dev:
            Lists of ``(undiac, diac)`` string pairs.  Only the diac
            (second) element is written to disk for CATT training.
        output_dir:
            Directory for checkpoints and logs.
        epochs:
            Number of training epochs.
        batch_size:
            Mini-batch size.
        tashkeel_threshold:
            Minimum diacritic-to-letter ratio; sentences below this are
            dropped by ``TashkeelDataset``.  Lower this if your data is
            sparsely diacritized.
        """
        import pytorch_lightning as pl
        from pytorch_lightning.callbacks import ModelCheckpoint
        from pytorch_lightning.callbacks.progress import TQDMProgressBar
        from pytorch_lightning.loggers import CSVLogger
        from tashkeel_dataset import TashkeelDataset, PrePaddingDataLoader

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_dir = tmp_path / "train"
            val_dir = tmp_path / "val"
            train_dir.mkdir()
            val_dir.mkdir()

            (train_dir / "train.txt").write_text(
                "\n".join(diac for _, diac in train), encoding="utf-8"
            )
            (val_dir / "val.txt").write_text(
                "\n".join(diac for _, diac in dev), encoding="utf-8"
            )

            train_dataset = TashkeelDataset(
                str(train_dir),
                self._tokenizer,
                self._max_seq_len,
                tashkeel_to_text_ratio_threshold=tashkeel_threshold,
            )
            val_dataset = TashkeelDataset(
                str(val_dir),
                self._tokenizer,
                self._max_seq_len,
                tashkeel_to_text_ratio_threshold=tashkeel_threshold,
            )

            train_loader = PrePaddingDataLoader(
                self._tokenizer, train_dataset,
                batch_size=batch_size, num_workers=0, shuffle=True,
            )
            val_loader = PrePaddingDataLoader(
                self._tokenizer, val_dataset,
                batch_size=batch_size, num_workers=0, shuffle=False,
            )

            ckpt_dir = Path(output_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_cb = ModelCheckpoint(
                dirpath=str(ckpt_dir),
                save_top_k=1,
                monitor="val_der",
                filename=f"catt_{self._model_type}" + "-{epoch:02d}-{val_der:.4f}",
            )

            trainer = pl.Trainer(
                accelerator="gpu" if torch.cuda.is_available() else "cpu",
                devices=1,
                max_epochs=epochs,
                callbacks=[TQDMProgressBar(refresh_rate=10), checkpoint_cb],
                logger=CSVLogger(save_dir=str(ckpt_dir / "logs")),
                enable_progress_bar=True,
            )

            trainer.fit(self._model, train_loader, val_loader)

            best = checkpoint_cb.best_model_path
            if best and Path(best).exists():
                self.load(best)

    def save(self, path: str) -> None:
        """Save the model's state dict to *path* (a ``.pt`` file)."""
        torch.save(self._model.state_dict(), path)

    def load(self, path: str) -> None:
        """Load a state dict from *path* (a ``.pt`` file) produced by ``save()``
        or one of the official CATT checkpoints from the GitHub releases:

        - ``best_eo_mlm_ns_epoch_193.pt`` (encoder-only, default)
        - ``best_ed_mlm_ns_epoch_178.pt`` (encoder-decoder)

        Download from: https://github.com/abjadai/catt/releases/tag/v2
        """
        state = torch.load(path, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state)
        self._model.eval()
