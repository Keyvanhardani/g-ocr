"""GOCR-Gewichte laden (von HuggingFace)."""
import os

from .pipeline import GOCR

# HF-Repo mit den GOCR-Gewichten (gocr_det.onnx, gocr_rec.onnx, charset.txt).
DEFAULT_HF_REPO = os.environ.get("GOCR_HF_REPO", "Keyven/g-ocr")


def from_pretrained(repo=None, det="gocr_det.onnx", rec="gocr_rec.onnx",
                    charset="charset.txt", **kwargs):
    """Lädt Detektor + Recognizer + Charset aus dem HF-Repo und baut die GOCR-Engine.

    kwargs werden an GOCR() durchgereicht (z. B. drop_score, num_threads).
    """
    from huggingface_hub import hf_hub_download
    repo = repo or DEFAULT_HF_REPO
    det_p = hf_hub_download(repo, det)
    rec_p = hf_hub_download(repo, rec)
    cs_p = hf_hub_download(repo, charset)
    return GOCR(det_p, rec_p, cs_p, **kwargs)
