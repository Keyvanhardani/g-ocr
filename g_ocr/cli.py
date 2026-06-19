"""GOCR CLI:  g-ocr datei.(png|jpg|webp|tiff|pdf ...)  ->  Text + bbox als JSON."""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(
        prog="g-ocr",
        description="GOCR — deutsche OCR: Dokument -> Text + Position (bbox). "
                    "Bilder (png/jpg/webp/tiff/bmp ...) und PDF.")
    ap.add_argument("path", help="Bild- oder PDF-Pfad")
    ap.add_argument("--repo", default=None, help="HF-Repo der GOCR-Gewichte")
    ap.add_argument("--det", help="Detektor-ONNX (statt --repo)")
    ap.add_argument("--rec", help="Recognizer-ONNX (statt --repo)")
    ap.add_argument("--charset", help="Charset-Datei (statt --repo)")
    ap.add_argument("--drop-score", type=float, default=0.4)
    ap.add_argument("--max-pages", type=int, default=500, help="PDF: max. Seiten")
    ap.add_argument("--dpi", type=int, default=200, help="PDF: Render-DPI")
    ap.add_argument("--text-only", action="store_true", help="nur Text ausgeben")
    a = ap.parse_args()

    from .pipeline import GOCR
    if a.det and a.rec and a.charset:
        ocr = GOCR(a.det, a.rec, a.charset, drop_score=a.drop_score)
    else:
        from .pretrained import from_pretrained
        ocr = from_pretrained(repo=a.repo, drop_score=a.drop_score)

    from .formats import is_pdf
    if is_pdf(a.path):
        res = ocr.read_document(a.path, max_pages=a.max_pages, dpi=a.dpi)
    else:
        res = ocr.read(a.path)

    # Beide Schemata haben ein Top-Level-"text" (Volltext / über alle Seiten).
    print(res["text"] if a.text_only
          else json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
