import tempfile, os
from pathlib import Path
from diac.data import load_sentences, make_pairs

SAMPLE = "فَإِنْ لَمْ يَكُونَا كَذَلِكَ\nقَالَ الْإِسْنَوِيُّ\n\n"

def test_load_sentences_strips_blanks(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(SAMPLE, encoding="utf-8")
    sents = load_sentences(f)
    assert len(sents) == 2
    assert sents[0] == "فَإِنْ لَمْ يَكُونَا كَذَلِكَ"

def test_make_pairs_strips_diacritics():
    sents = ["فَإِنْ"]
    pairs = make_pairs(sents)
    assert len(pairs) == 1
    undiac, diac = pairs[0]
    assert diac == "فَإِنْ"
    assert "َ" not in undiac  # fatha stripped
    assert "فإن" in undiac
