# g-ocr (Node)

**GOCR — schnelle, kleine OCR: Dokument → Text + Position (bbox) als JSON.** ~38 MB, reine CPU, kein GPU.
Node-Port des Python-Pakets [g-ocr](https://github.com/Keyvanhardani/g-ocr).

```bash
npm install g-ocr
```

```js
const gocr = require("g-ocr");

(async () => {
  const ocr = await gocr.fromPretrained();      // lädt die Gewichte von HuggingFace (gecached)
  const res = await ocr.read("dokument.png");   // png/jpg/webp/tiff/bmp ...
  console.log(res.text);
  // res = { engine, version, image:{width,height}, text,
  //         n_regions, regions:[{ id, text, score, box:[x0,y0,x1,y1], quad }] }
})();
```

CLI:

```bash
npx g-ocr dokument.png              # strukturiertes JSON (text + box + quad)
npx g-ocr dokument.png --text-only  # nur Text (Lesereihenfolge)
```

## Benchmarks (anerkannte Sets, CPU)

- **Scene-Text** Word Accuracy: IIIT5K **93,2 %**, ICDAR2013 **94,1 %** — klar vor EasyOCR (68,2 %).
- **Dokument-OCR** CER: **#2 von 5** (SROIE 18,9 / FUNSD 22,4) — vor EasyOCR, Tesseract & OCR.space, knapp hinter PaddleOCR.

🤗 Modell: <https://huggingface.co/Keyven/g-ocr> · 🖥️ Demo: <https://huggingface.co/spaces/Keyven/GOCR-Demo> · 🌐 <https://german-ocr.de>

## Lizenz

Apache-2.0
