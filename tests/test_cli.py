import json
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
    for key in ("DER", "DER*", "WER", "WER*"):
        assert key in result.output, f"Expected '{key}' in evaluate output"

def test_evaluate_mismatched_line_counts_exits_nonzero(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    # 1 undiacritized line → 1 hypothesis line; 2 reference lines → mismatch
    inp.write_text("فإن لم يكونا\n", encoding="utf-8")
    ref.write_text("فَإِنْ لَمْ يَكُونَا\nقَالَ\n", encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code != 0
    assert "hypothesis lines" in result.output

def test_finetune_camel_raises_gracefully(tmp_path):
    tr = tmp_path / "train.txt"
    dv = tmp_path / "dev.txt"
    tr.write_text(SAMPLE_DIAC, encoding="utf-8")
    dv.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["finetune", "--model", "camel",
                                  "--train", str(tr), "--dev", str(dv)])
    assert result.exit_code != 0
    assert "not support" in result.output.lower() or "not support" in str(result.exception).lower()

def test_evaluate_camel_format_json(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref),
                                  "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    for key in ("DER", "DER*", "WER", "WER*"):
        assert key in data
        assert isinstance(data[key], float)

def test_evaluate_camel_format_text_is_default(tmp_path):
    inp = tmp_path / "in.txt"
    ref = tmp_path / "ref.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    ref.write_text(SAMPLE_DIAC, encoding="utf-8")
    result = runner.invoke(app, ["evaluate", "--model", "camel",
                                  "--input", str(inp), "--ref", str(ref)])
    assert result.exit_code == 0
    # text format: each line is "KEY: value"
    assert "DER:" in result.output
    assert "WER:" in result.output

def test_infer_batch_size_flag(tmp_path):
    """--batch-size is accepted and does not break infer output."""
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out),
                                  "--batch-size", "4"])
    assert result.exit_code == 0
    assert out.exists()


def test_infer_env_var_checkpoint_does_not_crash_on_missing(tmp_path, monkeypatch):
    """If CATT_CHECKPOINT points to a nonexistent file, the CLI errors clearly
    rather than silently ignoring it. We test with camel (no env var) to confirm
    no spurious env-var load is attempted for backends without a mapping."""
    monkeypatch.delenv("CATT_CHECKPOINT", raising=False)
    monkeypatch.delenv("RABABA_CHECKPOINT", raising=False)
    inp = tmp_path / "in.txt"
    inp.write_text(SAMPLE_UNDIAC, encoding="utf-8")
    out = tmp_path / "out.txt"
    result = runner.invoke(app, ["infer", "--model", "camel",
                                  "--input", str(inp), "--output", str(out)])
    assert result.exit_code == 0
