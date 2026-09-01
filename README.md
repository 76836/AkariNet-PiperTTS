<p align="left">
  <img src="https://raw.githubusercontent.com/76836/Akari/main/images/banner.png" width="100%" alt="AkariNet"/>
</p>

> ### **This repository is a part of the [AkariNet](https://github.com/76836/Akari) project.**

# AkariNet-PiperTTS

Offline **Piper** text-to-speech for AkariNet. Runs entirely in the browser (ONNX Runtime Web + espeak-ng phonemizer).

Fine-tuned on Akari voice data. Compatible with the same `initTTS` / `speak` / `interrupt` pattern used by PocketTTS and KittenTTS.

## Quick start (standalone demo)

```bash
# After you upload the large binaries (see below)
python3 -m http.server 8000
# → open http://localhost:8000/
```

Or open `index.html` from `file://` (falls back to base64 copies if present).

## Files you must upload yourself

These are too large for automated writes. Upload them to the **root** of this repo (or release assets):

| File | Approx size | Notes |
|------|-------------|-------|
| `model.onnx` | ~61–64 MB | The Piper voice model (Akari fine-tune) |
| `ort-wasm.wasm` | ~22 MB | ONNX Runtime WASM binary |
| `phonemizer.js` | ~1.3 MB | Self-contained espeak-ng phonemizer |
| `ort.all.min.js` | ~0.8 MB | ORT UMD build (or use CDN — see below) |
| `ort-wasm.mjs` | ~45 KB | ORT WASM glue (already can be committed) |

**Recommended upload location:** root of `main` so GitHub Pages serves them at:

- `https://76836.github.io/AkariNet-PiperTTS/model.onnx`
- `https://76836.github.io/AkariNet-PiperTTS/ort-wasm.wasm`
- etc.

You can also use a Release and point the config at the release asset URLs.

## CDN options (recommended to keep repo light)

In `worker.js` / the Akari loader you can point ORT at the official CDN instead of shipping `ort.all.min.js`:

```js
// Example — use latest 1.22.x or pin the version you tested
importScripts("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.js");
```

`phonemizer.js` must stay self-hosted (it embeds the full espeak-ng data).

## AkariNet integration

1. Enable **GitHub Pages** on this repo (Settings → Pages → Deploy from branch `main` / root).
2. In the main [Akari](https://github.com/76836/Akari) repo the loader lives at:
   `characters/akari/PiperTTS.js`
3. Settings → Voice and the setup wizard offer “Akari voice (Piper offline)” which points at that loader.

The loader imports from `https://76836.github.io/AkariNet-PiperTTS/…` and wires lipsync the same way KittenTTS / PocketTTS do.

## Inference parameters

```js
const INFERENCE = {
  noise_scale: 0.667,  // pitch variation
  noise_w: 0.8,        // duration jitter — lower (0.4) = cleaner / less rough
};
```

## Credits

- [Piper TTS](https://github.com/OHF-Voice/piper1-gpl) (GPL)
- [ONNX Runtime Web](https://onnxruntime.ai/) (MIT)
- [phonemizer](https://www.npmjs.com/package/phonemizer) (MIT)
- AkariNet PocketTTS / KittenTTS patterns

GPL-licensed (inherited from Piper).
