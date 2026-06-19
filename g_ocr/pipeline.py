"""GOCR-Engine — Detektor (DB) + Recognizer (CTC), reines ONNX / CPU.

Eigenständige Inferenz-Pipeline (kein Fremd-OCR-Import):
  ocr = GOCR(det_onnx, rec_onnx, charset)
  ocr.read(img)            -> {engine, version, image, text, n_regions, regions[box, quad, score]}
  ocr.read_document(path)  -> mehrseitig (PDF/Bild): {n_pages, pages:[...], text}

Gewichte/Charset sind konfigurierbar (eigene Modelle werden hier eingehängt).
"""
import os
import warnings

import numpy as np
import cv2
import onnxruntime as ort

from .formats import load_image, is_pdf, iter_pdf_pages

try:
    import pyclipper
    from shapely.geometry import Polygon
    _HAS_CLIP = True
except Exception:  # pragma: no cover
    _HAS_CLIP = False

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


# ----------------------------- Charset / CTC -----------------------------
def load_charset(path):
    """CTC-Charset: index 0 = blank, dann Dict-Zeilen, am Ende Space."""
    chars = ["<blank>"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            chars.append(line.rstrip("\n"))
    chars.append(" ")
    return chars


def ctc_greedy_decode(probs, charset):
    """probs: [T, C] (softmax). Collapse-Repeats + Blank(0) entfernen."""
    idx = probs.argmax(1)
    conf = probs.max(1)
    out, confs, prev = [], [], -1
    for i, p in zip(idx, conf):
        if i != 0 and i != prev:
            if i < len(charset):
                out.append(charset[i])
                confs.append(p)
        prev = i
    return "".join(out), (float(np.mean(confs)) if confs else 0.0)


# ----------------------------- Detektor (DB) -----------------------------
def _det_preprocess(img_bgr, limit_side_len=960):
    h, w = img_bgr.shape[:2]
    ratio = min(1.0, limit_side_len / max(h, w))
    rh = max(32, int(round(h * ratio / 32)) * 32)
    rw = max(32, int(round(w * ratio / 32)) * 32)
    resized = cv2.resize(img_bgr, (rw, rh))
    x = resized.astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), (h, w, rh, rw)


def _order_quad(pts):
    """4 Punkte -> tl, tr, br, bl."""
    pts = pts[np.argsort(pts[:, 0])]
    left = pts[:2][np.argsort(pts[:2, 1])]
    right = pts[2:][np.argsort(pts[2:, 1])]
    return np.array([left[0], right[0], right[1], left[1]], np.float32)


def _unclip(box, ratio):
    poly = Polygon(box)
    if poly.length == 0:
        return None
    dist = poly.area * ratio / poly.length
    off = pyclipper.PyclipperOffset()
    off.AddPath([tuple(p) for p in box], pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    res = off.Execute(dist)
    return np.array(res[0]) if res else None


def _box_score_slow(prob, contour):
    h, w = prob.shape
    c = contour.reshape(-1, 2).copy()
    xmin = int(np.clip(c[:, 0].min(), 0, w - 1)); xmax = int(np.clip(c[:, 0].max(), 0, w - 1))
    ymin = int(np.clip(c[:, 1].min(), 0, h - 1)); ymax = int(np.clip(c[:, 1].max(), 0, h - 1))
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), np.uint8)
    c[:, 0] -= xmin; c[:, 1] -= ymin
    cv2.fillPoly(mask, [c.astype(np.int32)], 1)
    return cv2.mean(prob[ymin:ymax + 1, xmin:xmax + 1], mask)[0]


def _db_boxes(prob, shape, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5,
              max_candidates=1000, min_size=3):
    h0, w0, rh, rw = shape
    bitmap = (prob > thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes, scores = [], []
    for contour in contours[:max_candidates]:
        eps = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)
        if approx.shape[0] < 4:
            continue
        score = _box_score_slow(prob, contour)
        if score < box_thresh:
            continue
        ub = _unclip(approx, unclip_ratio)
        if ub is None or len(ub) < 4:
            continue
        rect = cv2.minAreaRect(ub.reshape(-1, 2).astype(np.float32))
        if min(rect[1]) < min_size:
            continue
        quad = _order_quad(cv2.boxPoints(rect))
        quad[:, 0] = np.clip(quad[:, 0] / rw * w0, 0, w0)
        quad[:, 1] = np.clip(quad[:, 1] / rh * h0, 0, h0)
        boxes.append(quad)
        scores.append(float(score))
    return boxes, scores


def _reading_order(boxes):
    """Sortiere Boxen oben→unten, links→rechts (Zeilen-tolerant)."""
    if not boxes:
        return []
    idx = list(range(len(boxes)))
    tops = [b[:, 1].min() for b in boxes]
    lefts = [b[:, 0].min() for b in boxes]
    heights = [max(1, b[:, 1].max() - b[:, 1].min()) for b in boxes]
    tol = np.median(heights) * 0.6
    idx.sort(key=lambda i: (round(tops[i] / max(tol, 1)), lefts[i]))
    return idx


def _crop(img, quad):
    quad = quad.astype(np.float32)
    wA = np.linalg.norm(quad[0] - quad[1]); wB = np.linalg.norm(quad[3] - quad[2])
    hA = np.linalg.norm(quad[0] - quad[3]); hB = np.linalg.norm(quad[1] - quad[2])
    W, H = int(max(wA, wB)), int(max(hA, hB))
    if W < 1 or H < 1:
        return None
    dst = np.array([[0, 0], [W, 0], [W, H], [0, H]], np.float32)
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(quad, dst), (W, H))
    if H * 1.0 / W >= 1.5:          # hohe Boxen drehen
        crop = np.rot90(crop)
    return crop


# ----------------------------- Recognizer (CTC) --------------------------
def _rec_preprocess(crop_bgr, img_h=48, max_w=320):
    h, w = crop_bgr.shape[:2]
    rw = min(max_w, max(1, int(round(img_h * w / max(h, 1)))))
    resized = cv2.resize(crop_bgr, (rw, img_h))
    x = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
    return x.transpose(2, 0, 1)[None].astype(np.float32)


# ----------------------------- GOCR ---------------------------------------
class GOCR:
    """Deutsche OCR-Engine: Detektor + Recognizer, reines ONNX/CPU."""

    def __init__(self, det_onnx, rec_onnx, charset, drop_score=0.4,
                 limit_side_len=960, rec_h=48, rec_max_w=2000, num_threads=None):
        so = ort.SessionOptions()
        if num_threads:
            so.intra_op_num_threads = int(num_threads)
        prov = ["CPUExecutionProvider"]
        self.det = ort.InferenceSession(det_onnx, sess_options=so, providers=prov)
        self.rec = ort.InferenceSession(rec_onnx, sess_options=so, providers=prov)
        self.det_in = self.det.get_inputs()[0].name
        self.rec_in = self.rec.get_inputs()[0].name
        self.charset = charset if isinstance(charset, list) else load_charset(charset)
        self.drop_score = drop_score
        self.limit_side_len = limit_side_len
        self.rec_h, self.rec_max_w = rec_h, rec_max_w

    def detect(self, img_bgr):
        x, shape = _det_preprocess(img_bgr, self.limit_side_len)
        prob = self.det.run(None, {self.det_in: x})[0][0, 0]
        boxes, _ = _db_boxes(prob, shape)
        return [boxes[i] for i in _reading_order(boxes)]

    def recognize(self, crop_bgr):
        x = _rec_preprocess(crop_bgr, self.rec_h, self.rec_max_w)
        probs = self.rec.run(None, {self.rec_in: x})[0][0]
        return ctc_greedy_decode(probs, self.charset)

    def _to_bgr(self, image):
        """Pfad (Bild) oder numpy -> BGR-numpy. PDF: siehe read_document()."""
        if isinstance(image, str):
            return load_image(image)             # robust inkl. webp (Pillow)
        arr = np.asarray(image)
        if arr.ndim == 2:                        # Graustufen -> 3 Kanäle
            arr = np.stack([arr, arr, arr], -1)
        if arr.shape[2] == 3:                    # RGB -> BGR (Annahme)
            arr = arr[:, :, ::-1]
        return np.ascontiguousarray(arr)

    def _ocr_page(self, img_bgr):
        """BGR-numpy -> {image, text, n_regions, regions} (eine Seite)."""
        h, w = img_bgr.shape[:2]
        regions = []
        for quad in self.detect(img_bgr):
            crop = _crop(img_bgr, quad)
            if crop is None:
                continue
            text, score = self.recognize(crop)
            if not text or score < self.drop_score:
                continue
            xs = [int(p[0]) for p in quad]
            ys = [int(p[1]) for p in quad]
            regions.append({
                "id": len(regions),
                "text": text,
                "score": round(score, 3),
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "quad": [[int(x), int(y)] for x, y in quad],
            })
        return {
            "image": {"width": int(w), "height": int(h)},
            "text": "\n".join(r["text"] for r in regions),
            "n_regions": len(regions),
            "regions": regions,
        }

    def read(self, image):
        """Ein Bild (Pfad/numpy) -> strukturiertes GOCR-JSON (eine Seite).

        Schema: {engine, version, image:{width,height}, text, n_regions,
                 regions:[{id, text, score, box:[x0,y0,x1,y1], quad:[[x,y]x4]}]}
          - box  = achsenparalleles Rechteck (kompakt)   - quad = 4 Eckpunkte
          - text = Volltext in Lesereihenfolge (direkt fürs LLM)
        Bildformate inkl. webp/tiff/bmp. Für PDF/mehrseitig: read_document().
        """
        page = self._ocr_page(self._to_bgr(image))
        return {"engine": "GOCR", "version": "0.2.0", **page}

    def read_document(self, source, max_pages=500, dpi=200):
        """Bild ODER PDF (Pfad) -> Dokument-JSON über alle Seiten.

        Schema: {engine, version, source, n_pages, n_pages_total, truncated,
                 text, pages:[{page, image, text, n_regions, regions}]}
        PDF-Support ist ein optionales Plugin:  pip install g-ocr[pdf].
        max_pages begrenzt sehr große PDFs (Default 500); truncated +
        n_pages_total zeigen ehrlich, ob abgeschnitten wurde.
        """
        pages = []
        n_total = 1
        if is_pdf(source):
            for i, total, bgr in iter_pdf_pages(source, dpi=dpi, max_pages=max_pages):
                n_total = total
                pages.append({"page": i + 1, **self._ocr_page(bgr)})
        else:
            pages.append({"page": 1, **self._ocr_page(self._to_bgr(source))})
        truncated = n_total > len(pages)
        if truncated:
            warnings.warn(
                f"GOCR: nur {len(pages)}/{n_total} Seiten verarbeitet "
                f"(max_pages={max_pages}).")
        return {
            "engine": "GOCR",
            "version": "0.2.0",
            "source": source if isinstance(source, str) else "<array>",
            "n_pages": len(pages),
            "n_pages_total": n_total,
            "truncated": truncated,
            "text": "\n\n".join(p["text"] for p in pages),
            "pages": pages,
        }


def read(image, det_onnx, rec_onnx, charset, **kw):
    """Komfort-Funktion (ein Bild)."""
    return GOCR(det_onnx, rec_onnx, charset, **kw).read(image)


def read_document(source, det_onnx, rec_onnx, charset, *, max_pages=500, dpi=200, **kw):
    """Komfort-Funktion (Bild/PDF, mehrseitig)."""
    return GOCR(det_onnx, rec_onnx, charset, **kw).read_document(
        source, max_pages=max_pages, dpi=dpi)
