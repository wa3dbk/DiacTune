from __future__ import annotations
import tempfile
from pathlib import Path
from diacritization_evaluation import der, wer


def compute_der(hyp: list[str], ref: list[str]) -> dict[str, float]:
    """Compute DER and WER between hypothesis and reference sentence lists.

    Writes temporary files because diacritization_evaluation operates on paths.
    """
    with tempfile.TemporaryDirectory() as tmp:
        orig_path = str(Path(tmp) / "ref.txt")
        pred_path = str(Path(tmp) / "hyp.txt")
        Path(orig_path).write_text("\n".join(ref), encoding="utf-8")
        Path(pred_path).write_text("\n".join(hyp), encoding="utf-8")
        return {
            "DER":  der.calculate_der_from_path(orig_path, pred_path),
            "DER*": der.calculate_der_from_path(orig_path, pred_path, case_ending=False),
            "WER":  wer.calculate_wer_from_path(orig_path, pred_path),
            "WER*": wer.calculate_wer_from_path(orig_path, pred_path, case_ending=False),
        }
