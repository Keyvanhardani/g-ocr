"use strict";
/**
 * GOCR (Node) — Detektor (DB) + Recognizer (CTC), reines ONNX / CPU.
 * Node-Port der g-ocr Python-Pipeline. Gewichte werden von HuggingFace geladen.
 *
 *   const gocr = require("g-ocr");
 *   const ocr = await gocr.fromPretrained();
 *   const res = await ocr.read("dokument.png");   // {text, regions:[{text, box, quad, score}]}
 */
const ort = require("onnxruntime-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
const os = require("os");

const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];
const DEFAULT_REPO = process.env.GOCR_HF_REPO || "Keyven/g-ocr";

function cacheDir() {
  const d = path.join(os.homedir(), ".cache", "g-ocr");
  fs.mkdirSync(d, { recursive: true });
  return d;
}

async function ensureFile(repo, name) {
  const dest = path.join(cacheDir(), name);
  if (fs.existsSync(dest) && fs.statSync(dest).size > 0) return dest;
  const url = `https://huggingface.co/${repo}/resolve/main/${name}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`GOCR: Download fehlgeschlagen (${res.status}) ${url}`);
  fs.writeFileSync(dest, Buffer.from(await res.arrayBuffer()));
  return dest;
}

function loadCharset(file) {
  let lines = fs.readFileSync(file, "utf-8").split("\n");
  if (lines.length && lines[lines.length - 1] === "") lines.pop(); // trailing newline
  lines = lines.map((l) => (l.endsWith("\r") ? l.slice(0, -1) : l));
  return ["<blank>", ...lines, " "]; // 0 = CTC-Blank, Ende = Space
}

// ---- Detektor-Preprocess: BGR, ImageNet-Norm (wie Python) ----
async function detPreprocess(imgPath, limit) {
  const meta = await sharp(imgPath).metadata();
  const w0 = meta.width, h0 = meta.height;
  const ratio = Math.min(1.0, limit / Math.max(w0, h0));
  const rh = Math.max(32, Math.round((h0 * ratio) / 32) * 32);
  const rw = Math.max(32, Math.round((w0 * ratio) / 32) * 32);
  const { data } = await sharp(imgPath).removeAlpha().resize(rw, rh, { fit: "fill" })
    .raw().toBuffer({ resolveWithObject: true }); // RGB
  const HW = rh * rw;
  const chw = new Float32Array(3 * HW);
  for (let p = 0; p < HW; p++) {
    const i = p * 3;
    const b = data[i + 2] / 255, g = data[i + 1] / 255, r = data[i] / 255; // -> BGR
    chw[0 * HW + p] = (b - MEAN[0]) / STD[0];
    chw[1 * HW + p] = (g - MEAN[1]) / STD[1];
    chw[2 * HW + p] = (r - MEAN[2]) / STD[2];
  }
  return { tensor: new ort.Tensor("float32", chw, [1, 3, rh, rw]), w0, h0, rw, rh };
}

// ---- DB-Postprocess (pure JS): threshold -> hor. Dilation -> Connected Components -> Boxen ----
function dbBoxes(prob, rw, rh, w0, h0, thr, dil, minSize) {
  const bin = new Uint8Array(rw * rh);
  for (let i = 0; i < prob.length; i++) bin[i] = prob[i] > thr ? 1 : 0;
  // horizontale Dilation (verbindet Zeichen zu Wörtern/Zeilen)
  const dl = new Uint8Array(rw * rh);
  for (let y = 0; y < rh; y++) {
    const row = y * rw;
    for (let x = 0; x < rw; x++) {
      if (!bin[row + x]) continue;
      const x0 = Math.max(0, x - dil), x1 = Math.min(rw - 1, x + dil);
      for (let xx = x0; xx <= x1; xx++) dl[row + xx] = 1;
    }
  }
  // Connected Components (BFS, 8-Nachbarschaft)
  const seen = new Uint8Array(rw * rh);
  const boxes = [];
  const stack = new Int32Array(rw * rh);
  for (let s = 0; s < rw * rh; s++) {
    if (!dl[s] || seen[s]) continue;
    let sp = 0; stack[sp++] = s; seen[s] = 1;
    let minx = rw, miny = rh, maxx = 0, maxy = 0, area = 0;
    while (sp > 0) {
      const c = stack[--sp];
      const cy = (c / rw) | 0, cx = c - cy * rw;
      if (cx < minx) minx = cx; if (cx > maxx) maxx = cx;
      if (cy < miny) miny = cy; if (cy > maxy) maxy = cy;
      area++;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const nx = cx + dx, ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= rw || ny >= rh) continue;
        const ni = ny * rw + nx;
        if (dl[ni] && !seen[ni]) { seen[ni] = 1; stack[sp++] = ni; }
      }
    }
    const bw = maxx - minx + 1, bh = maxy - miny + 1;
    if (Math.min(bw, bh) < minSize || area < minSize * minSize) continue;
    // leichte Aufweitung (unclip-Approx) + Skalierung auf Originalgröße
    const padx = Math.round(bh * 0.15), pady = Math.round(bh * 0.12);
    const X0 = Math.max(0, (minx - padx)) * (w0 / rw);
    const Y0 = Math.max(0, (miny - pady)) * (h0 / rh);
    const X1 = Math.min(rw, (maxx + padx + 1)) * (w0 / rw);
    const Y1 = Math.min(rh, (maxy + pady + 1)) * (h0 / rh);
    boxes.push([Math.round(X0), Math.round(Y0), Math.round(X1), Math.round(Y1)]);
  }
  return boxes;
}

function readingOrder(boxes) {
  if (!boxes.length) return [];
  const heights = boxes.map((b) => Math.max(1, b[3] - b[1]));
  const sorted = [...heights].sort((a, b) => a - b);
  const tol = Math.max(1, sorted[(sorted.length / 2) | 0] * 0.6);
  return boxes.map((b, i) => i).sort((i, j) => {
    const ri = Math.round(boxes[i][1] / tol), rj = Math.round(boxes[j][1] / tol);
    return ri !== rj ? ri - rj : boxes[i][0] - boxes[j][0];
  });
}

// ---- Recognizer-Preprocess: BGR, (x/255-0.5)/0.5 ----
async function recPreprocess(imgPath, box, recH, maxW) {
  let [x0, y0, x1, y1] = box;
  const bw = Math.max(1, x1 - x0), bh = Math.max(1, y1 - y0);
  let pipe = sharp(imgPath).extract({ left: x0, top: y0, width: bw, height: bh });
  let cw = bw, ch = bh;
  if (ch / cw >= 1.5) { pipe = pipe.rotate(90); [cw, ch] = [ch, cw]; } // hohe Box drehen
  const rw = Math.min(maxW, Math.max(1, Math.round((recH * cw) / ch)));
  const { data } = await pipe.removeAlpha().resize(rw, recH, { fit: "fill" })
    .raw().toBuffer({ resolveWithObject: true }); // RGB
  const HW = recH * rw;
  const chw = new Float32Array(3 * HW);
  for (let p = 0; p < HW; p++) {
    const i = p * 3;
    chw[0 * HW + p] = (data[i + 2] / 255 - 0.5) / 0.5; // B
    chw[1 * HW + p] = (data[i + 1] / 255 - 0.5) / 0.5; // G
    chw[2 * HW + p] = (data[i] / 255 - 0.5) / 0.5;     // R
  }
  return new ort.Tensor("float32", chw, [1, 3, recH, rw]);
}

function ctcDecode(probs, T, C, charset) {
  let out = "", confs = 0, n = 0, prev = -1;
  for (let t = 0; t < T; t++) {
    let best = 0, bestv = -Infinity;
    const base = t * C;
    for (let c = 0; c < C; c++) { const v = probs[base + c]; if (v > bestv) { bestv = v; best = c; } }
    if (best !== 0 && best !== prev && best < charset.length) { out += charset[best]; confs += bestv; n++; }
    prev = best;
  }
  return { text: out, score: n ? confs / n : 0 };
}

class GOCR {
  constructor(det, rec, charset, opts = {}) {
    this.det = det; this.rec = rec; this.charset = charset;
    this.dropScore = opts.dropScore ?? 0.4;
    this.limit = opts.limitSideLen ?? 960;
    this.recH = opts.recH ?? 48;
    this.recMaxW = opts.recMaxW ?? 2000;
    this.thr = opts.detThresh ?? 0.3;
    this.dilate = opts.dilate ?? 4;
    this.minSize = opts.minSize ?? 3;
  }

  async read(imagePath) {
    const { tensor, w0, h0, rw, rh } = await detPreprocess(imagePath, this.limit);
    const detOut = await this.det.run({ [this.det.inputNames[0]]: tensor });
    const probT = detOut[this.det.outputNames[0]];
    const prob = probT.data; // [1,1,rh,rw]
    let boxes = dbBoxes(prob, rw, rh, w0, h0, this.thr, this.dilate, this.minSize);
    const order = readingOrder(boxes);
    const regions = [];
    for (const idx of order) {
      const box = boxes[idx];
      let recT;
      try { recT = await recPreprocess(imagePath, box, this.recH, this.recMaxW); }
      catch (e) { continue; }
      const recOut = await this.rec.run({ [this.rec.inputNames[0]]: recT });
      const o = recOut[this.rec.outputNames[0]];
      const dims = o.dims; // [1, T, C]
      const T = dims[1], C = dims[2];
      const { text, score } = ctcDecode(o.data, T, C, this.charset);
      if (!text || score < this.dropScore) continue;
      const [X0, Y0, X1, Y1] = box;
      regions.push({
        id: regions.length, text, score: Math.round(score * 1000) / 1000,
        box: [X0, Y0, X1, Y1],
        quad: [[X0, Y0], [X1, Y0], [X1, Y1], [X0, Y1]],
      });
    }
    return {
      engine: "GOCR", version: "0.2.0",
      image: { width: w0, height: h0 },
      text: regions.map((r) => r.text).join("\n"),
      n_regions: regions.length, regions,
    };
  }
}

async function fromPretrained(opts = {}) {
  const repo = opts.repo || DEFAULT_REPO;
  const [detP, recP, charP] = await Promise.all([
    ensureFile(repo, opts.det || "gocr_det.onnx"),
    ensureFile(repo, opts.rec || "gocr_rec.onnx"),
    ensureFile(repo, opts.charset || "charset.txt"),
  ]);
  const [det, rec] = await Promise.all([
    ort.InferenceSession.create(detP), ort.InferenceSession.create(recP),
  ]);
  return new GOCR(det, rec, loadCharset(charP), opts);
}

module.exports = { GOCR, fromPretrained, DEFAULT_REPO };
