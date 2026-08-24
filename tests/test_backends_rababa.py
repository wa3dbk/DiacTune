import pytest


def _check_rababa_env():
    """Return skip reason string if the environment can't run Rababa tests."""
    try:
        from rababa.config import load_task_config
        load_task_config("rababa_arabic")
    except Exception as exc:
        return f"rababa_arabic config unavailable: {exc}"
    try:
        from rababa.models.modern import build_modern_student  # noqa: F401
    except Exception as exc:
        return f"rababa model not importable: {exc}"
    try:
        # Verify swiglu module is present — it's lazily imported during
        # forward pass and may be missing from incomplete installations.
        from rababa.models import swiglu as _swiglu  # noqa: F401
    except Exception as exc:
        return f"rababa.models.swiglu not available: {exc}"
    try:
        from diac.backends.rababa import RababaBackend  # noqa: F401
    except Exception as exc:
        return f"RababaBackend not importable: {exc}"
    return None


_ENV_SKIP = _check_rababa_env()

pytestmark = pytest.mark.skipif(
    _ENV_SKIP is not None,
    reason=f"Rababa environment not compatible: {_ENV_SKIP}",
)

SAMPLE = ["فإن لم يكونا", "قال الإسنوي"]


def test_infer_returns_correct_length():
    from diac.backends.rababa import RababaBackend
    b = RababaBackend()  # no checkpoint → random weights, but shape correct
    results = b.infer(SAMPLE)
    assert len(results) == len(SAMPLE)


def test_infer_output_nonempty():
    from diac.backends.rababa import RababaBackend
    b = RababaBackend()
    result = b.infer(["كتب"])[0]
    assert len(result) > 0


def test_save_load_roundtrip(tmp_path):
    from diac.backends.rababa import RababaBackend
    b = RababaBackend()
    b.save(str(tmp_path / "rababa.pt"))
    b2 = RababaBackend()
    b2.load(str(tmp_path / "rababa.pt"))
    assert b2.infer(["كتب"])[0]  # just check it runs
