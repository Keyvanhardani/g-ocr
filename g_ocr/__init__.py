"""GOCR — schnelle, kleine deutsche OCR-/Vision-Schicht (CPU, kein GPU).

Ganzes Dokument -> Text + Position (bbox) als strukturiertes JSON.
Bilder (png/jpg/webp/tiff/bmp ...) und PDF (Plugin: pip install g-ocr[pdf]).

    import g_ocr
    ocr = g_ocr.from_pretrained()               # lädt die GOCR-Gewichte (HF)
    res = ocr.read("dokument.png")              # eine Seite
    doc = ocr.read_document("rechnung.pdf")     # mehrseitig (PDF/Bild)
    # -> {text, regions:[{text, box:[x0,y0,x1,y1], quad, score}], ...}
"""
__version__ = "0.2.0"

from .pipeline import GOCR, read, read_document          # noqa: F401
from .pretrained import from_pretrained, DEFAULT_HF_REPO  # noqa: F401

__all__ = ["GOCR", "read", "read_document", "from_pretrained",
           "DEFAULT_HF_REPO", "__version__"]
