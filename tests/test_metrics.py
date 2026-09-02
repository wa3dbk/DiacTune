import pytest
from diac.metrics import compute_der

PERFECT_HYP = ["فَإِنْ لَمْ"]
PERFECT_REF = ["فَإِنْ لَمْ"]
WRONG_HYP   = ["فإن لم"]   # no diacritics at all

def test_perfect_prediction_gives_zero_der():
    result = compute_der(PERFECT_HYP, PERFECT_REF)
    assert result["DER"] == 0.0
    assert result["WER"] == 0.0

def test_missing_diacritics_gives_nonzero_der():
    result = compute_der(WRONG_HYP, PERFECT_REF)
    assert result["DER"] > 0.0
    assert result["WER"] > 0.0

def test_returns_all_keys():
    result = compute_der(PERFECT_HYP, PERFECT_REF)
    assert set(result.keys()) == {"DER", "DER*", "WER", "WER*"}

def test_compute_der_raises_on_length_mismatch():
    hyp = ["فَإِنْ لَمْ يَكُونَا"]
    ref = ["فَإِنْ لَمْ يَكُونَا", "قَالَ"]
    with pytest.raises(ValueError, match="hyp and ref must have the same number of lines"):
        compute_der(hyp, ref)
