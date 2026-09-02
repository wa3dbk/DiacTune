import pytest
from diac.backends.camel import CAMeLBackend


@pytest.fixture(scope="module")
def backend():
    return CAMeLBackend()


def test_infer_returns_same_length(backend):
    sentences = ["فإن لم يكونا", "قال الإسنوي"]
    results = backend.infer(sentences)
    assert len(results) == 2


def test_infer_adds_diacritics(backend):
    result = backend.infer(["كتب"])[0]
    # result should contain at least one diacritic character
    arabic_diacritics = set("ًٌٍَُِّْ")
    assert any(c in arabic_diacritics for c in result)


def test_finetune_raises(backend):
    with pytest.raises(NotImplementedError):
        backend.finetune([], [])


def test_camel_backend_satisfies_protocol():
    from diac.backends.base import DiacritizationProtocol
    from diac.backends.camel import CAMeLBackend
    b = CAMeLBackend()
    assert isinstance(b, DiacritizationProtocol)
