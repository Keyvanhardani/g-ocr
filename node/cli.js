#!/usr/bin/env node
"use strict";
const { fromPretrained } = require("./index");

async function main() {
  const args = process.argv.slice(2);
  const textOnly = args.includes("--text-only");
  const file = args.find((a) => !a.startsWith("--"));
  if (!file) {
    console.error("Usage: g-ocr <bild.(png|jpg|webp|tiff|bmp)> [--text-only]");
    process.exit(1);
  }
  const ocr = await fromPretrained();
  const res = await ocr.read(file);
  console.log(textOnly ? res.text : JSON.stringify(res, null, 2));
}

main().catch((e) => {
  console.error("GOCR:", e.message);
  process.exit(1);
});
