"""Tests for CATTBackend.

Skip guards
-----------
All tests are skipped automatically if:
- ``codes/catt/`` has not been cloned, or
- required dependencies (``pytorch_lightning``, ``kaldialign``) are missing.

Checkpoint-dependent accuracy tests
------------------------------------
Set the ``CATT_CHECKPOINT`` environment variable to the path of a downloaded
``.pt`` checkpoint to enable realistic inference.  Without it the model runs
with random weights (structure / shape tests only).

Download checkpoints from: https://github.com/abjadai/catt/releases/tag/v2
  - best_eo_mlm_ns_epoch_193.pt  (encoder-only, default)
  - best_ed_mlm_ns_epoch_178.pt  (encoder-decoder)
"""

import os

import pytest

from diac.backends.catt import CATT_UNAVAILABLE

# ---------------------------------------------------------------------------
# Module-level skip guard
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(
        CATT_UNAVAILABLE is not None,
        reason=f"CATT environment not available: {CATT_UNAVAILABLE}",
    ),
    pytest.mark.slow,
]

# Optional checkpoint path
CKPT = os.environ.get("CATT_CHECKPOINT", None)

SAMPLE = ["فإن لم يكونا", "قال"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def backend():
    """Return a CATTBackend, loading CKPT if provided."""
    from diac.backends.catt import CATTBackend

    b = CATTBackend()
    if CKPT:
        b.load(CKPT)
    return b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_infer_length(backend):
    """infer() must return the same number of strings as input sentences."""
    results = backend.infer(SAMPLE)
    assert len(results) == len(SAMPLE)


def test_infer_nonempty(backend):
    """infer() must return a non-empty string for a non-empty Arabic input."""
    result = backend.infer(["كتب"])[0]
    assert len(result) > 0


def test_infer_returns_strings(backend):
    """Every element of the infer() output must be a str."""
    results = backend.infer(SAMPLE)
    for r in results:
        assert isinstance(r, str)


def test_save_load_roundtrip(backend, tmp_path):
    """save() + load() must produce a functional model."""
    ckpt = str(tmp_path / "catt.pt")
    backend.save(ckpt)

    from diac.backends.catt import CATTBackend

    b2 = CATTBackend()
    b2.load(ckpt)
    result = b2.infer(["كتب"])[0]
    assert isinstance(result, str) and len(result) > 0


def test_default_model_type_is_eo():
    """CATTBackend() defaults to the encoder-only model."""
    from diac.backends.catt import CATTBackend

    b = CATTBackend()
    assert b._model_type == "eo"


def test_invalid_model_type_raises():
    """Passing an unrecognised model_type must raise ValueError."""
    from diac.backends.catt import CATTBackend

    with pytest.raises(ValueError, match="model_type"):
        CATTBackend(model_type="bad")
