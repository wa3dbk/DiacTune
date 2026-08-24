import tempfile
from pathlib import Path
from typer.testing import CliRunner
from diac.cli import app

runner = CliRunner()

SAMPLE_DIAC = "فَإِنْ لَمْ يَكُونَا\n"
SAMPLE_UNDIAC = "فإن لم يكونا\n"

def test_infer_camel(tmp_path):
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out)])
    assert result.exit_code == 0
    assert out.exists()

def test_evaluate_camel(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code == 0
    assert "DER" in result.output

def test_finetune_camel_raises_gracefully(tmp_path):
    tr = tmp_path / "train.txt"
    dv = tmp_path / "dev.txt"
    tr.write_text(SAMPLE_DIAC, encoding="utf-8")
    dv.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["finetune", "--model", "camel",
                                  "--train", str(tr), "--dev", str(dv)])
    assert result.exit_code != 0
    assert "not support" in result.output.lower() or "not support" in str(result.exception).lower()
