/**
 * AkariNet-PiperTTS — Web Worker for non-blocking speech generation.
 *
 * Loads ORT + phonemizer via importScripts (relative to this file's URL),
 * receives the model bytes from the main thread, and streams audio chunks
 * back per sentence.
 */

// Prefer CDN for ORT if you want a lighter repo; otherwise keep local copies.
importScripts(
  // "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.all.min.js",
  "ort.all.min.js",
  "phonemizer.js"
);

let session = null;
let config = null;
let generating = false;

function phonemesToIds(ipa, idMap) {
  const ids = [];
  ids.push(...idMap["^"], ...idMap["_"]);
  for (const ch of ipa.trim().normalize("NFD")) {
    if (ch in idMap) ids.push(...idMap[ch], ...idMap["_"]);
  }
  ids.push(...idMap["$"]);
  return ids;
}

self.onmessage = async (e) => {
  const { type } = e.data;

  if (type === "init") {
    const { modelBytes, config: cfg } = e.data;
    config = cfg;
    try {
      const base = self.location.href.replace(/[^/]*$/, "");
      self.ort.env.wasm.numThreads = 1;
      self.ort.env.wasm.wasmPaths = {
        mjs: base + "ort-wasm.mjs",
        wasm: base + "ort-wasm.wasm"
      };

      session = await self.ort.InferenceSession.create(modelBytes, {
        executionProviders: [{ name: "wasm", simd: true }]
      });
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
    return;
  }

  if (type === "speak") {
    generating = true;
    const { text, lengthScale, noise_scale, noise_w } = e.data;

    try {
      const sentences = await self.phonemize(text, config.espeak.voice);
      const idMap = config.phoneme_id_map;
      const total = sentences.length;

      for (let i = 0; i < total; i++) {
        if (!generating) break;

        const ipa = sentences[i];
        if (!ipa || !ipa.trim()) continue;
        const ids = phonemesToIds(ipa, idMap);
        if (ids.length <= 2) continue;

        const feeds = {
          input: new self.ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
          input_lengths: new self.ort.Tensor("int64", BigInt64Array.from([BigInt(ids.length)]), [1]),
          scales: new self.ort.Tensor("float32", Float32Array.from([noise_scale, lengthScale, noise_w]), [3])
        };
        const out = await session.run(feeds);

        const audio = out.output.data;
        self.postMessage({ type: "chunk", audio, sentence: i + 1, total }, [audio.buffer]);
      }

      self.postMessage({ type: "done" });
    } catch (err) {
      self.postMessage({ type: "error", message: err.message });
    }
    generating = false;
    return;
  }

  if (type === "stop") {
    generating = false;
    return;
  }
};
