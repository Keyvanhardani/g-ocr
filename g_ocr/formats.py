"""GOCR-Eingabeformate — pluggbare Loader für Bilder und PDF.

Bilder (png/jpg/webp/tiff/bmp/gif ...) laufen über Pillow (Core-Dep).
PDF ist ein **optionales Plugin**:  pip install g-ocr[pdf]  (pypdfium2).

Erweiterbar: weitere Formate via LOADERS-Registry (Endung -> Loader-Funktion,
die ein BGR-numpy-Array bzw. einen Seiten-Generator liefert).
"""
import os
import numpy as np

# Vom Pillow-Decoder abgedeckte Einzelbild-Formate.
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff",
             ".bmp", ".gif", ".ppm", ".pgm", ".jp2"}
PDF_EXTS = {".pdf"}


def _ext(path):
    return os.path.splitext(path)[1].lower() if isinstance(path, str) else ""


def _pil_to_bgr(im):
    """PIL-Image -> BGR-numpy (wie cv2.imread, RGB->BGR)."""
    arr = np.asarray(im.convert("RGB"))
    return np.ascontiguousarray(arr[:, :, ::-1])


def load_image(path):
    """Bild-Datei -> BGR-numpy. Robust inkl. webp/tiff/bmp/gif (Pillow)."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            return _pil_to_bgr(im)
    except Exception as e:                       # klare Fehlermeldung statt None
        raise ValueError(f"GOCR: Bild nicht lesbar: {path} ({e})") from e


def is_pdf(path):
    return _ext(path) in PDF_EXTS


def is_image(path):
    return _ext(path) in IMAGE_EXTS


def iter_pdf_pages(path, dpi=200, max_pages=500):
    """PDF -> Generator je Seite:  (index, total_seiten, bgr_numpy).

    Optionales Plugin:  pip install g-ocr[pdf]  (pypdfium2).
    Sehr große PDFs werden bei max_pages begrenzt (total wird trotzdem gemeldet).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError as e:
        raise ImportError(
            "GOCR: PDF-Support benötigt das optionale Plugin 'pypdfium2'.\n"
            "      Installieren:  pip install g-ocr[pdf]"
        ) from e

    pdf = pdfium.PdfDocument(path)
    try:
        total = len(pdf)
        n = min(total, max(0, int(max_pages)))
        scale = dpi / 72.0                       # PDF-Basis = 72 DPI
        for i in range(n):
            page = pdf[i]
            bitmap = page.render(scale=scale)
            pil = bitmap.to_pil()
            yield i, total, _pil_to_bgr(pil)
            page.close()
    finally:
        pdf.close()


# Registry: Endung -> ("image"|"pdf"). Für künftige Formate hier erweitern.
LOADERS = {**{e: "image" for e in IMAGE_EXTS}, **{e: "pdf" for e in PDF_EXTS}}
