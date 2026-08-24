import pytest

# Guard: transformers 5.x requires torch >= 2.5 to load models.
# If the environment does not satisfy this, skip all tests gracefully.
def _check_env():
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return f"torch not importable: {exc}"
    try:
        # Try actually instantiating an AutoModelForSeq2SeqLM placeholder to
        # confirm transformers considers torch available for model loading.
        from transformers import AutoModelForSeq2SeqLM
        # Access .from_pretrained — if torch is disabled, transformers raises
        # ImportError on attribute access of the placeholder class.
        _ = AutoModelForSeq2SeqLM.from_pretrained  # type: ignore[attr-defined]
    except Exception as exc:
        return str(exc)
    return None

_ENV_SKIP = _check_env()

pytestmark = pytest.mark.skipif(
    _ENV_SKIP is not None,
    reason=f"torch/transformers environment not compatible: {_ENV_SKIP}",
)


@pytest.fixture(scope="module")
def backend():
    from diac.backends.byt5 import ByT5Backend

    b = ByT5Backend()  # loads pretrained from HF Hub
    return b


def test_infer_returns_same_length(backend):
    results = backend.infer(["فإن لم يكونا", "قال"])
    assert len(results) == 2


def test_infer_output_contains_arabic(backend):
    result = backend.infer(["كتب"])[0]
    assert any("\u0600" <= c <= "\u06ff" for c in result)


def test_save_load_roundtrip(backend, tmp_path):
    backend.save(str(tmp_path / "byt5_ckpt"))
    from diac.backends.byt5 import ByT5Backend

    b2 = ByT5Backend.__new__(ByT5Backend)
    b2.load(str(tmp_path / "byt5_ckpt"))
    result = b2.infer(["كتب"])
    assert len(result) == 1
