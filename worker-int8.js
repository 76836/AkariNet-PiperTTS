/**
 * AkariNet-PiperTTS int8 worker.
 * Uses onnxruntime-web 1.30 from jsDelivr so quantized ops (ConvInteger, etc.)
 * are available — the older Pages ORT 1.22 build lacks those kernels.
 */
importScripts(
  "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.js",
  "phonemizer.js"
);

var session = null;
var config = null;
var generating = false;

function phonemesToIds(ipa, idMap) {
  var ids = [];
  ids.push.apply(ids, idMap["^"]);
  ids.push.apply(ids, idMap["_"]);
  var norm = ipa.trim().normalize("NFD");
  for (var i = 0; i < norm.length; i++) {
    var ch = norm[i];
    if (ch in idMap) {
      ids.push.apply(ids, idMap[ch]);
      ids.push.apply(ids, idMap["_"]);
    }
  }
  ids.push.apply(ids, idMap["$"]);
  return ids;
}

async function createSession(modelBytes) {
  self.ort.env.wasm.numThreads = 1;
  // Point at matching 1.30 WASM artifacts on the CDN
  self.ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

  var providers = [
    { name: "wasm", simd: true }
  ];

  // Try WASM first (CPU). If the build still rejects the graph, surface the error.
  try {
    return await self.ort.InferenceSession.create(modelBytes, {
      executionProviders: providers
    });
  } catch (e1) {
    // Optional WebGPU path on Chromium if WASM still lacks a kernel
    try {
      return await self.ort.InferenceSession.create(modelBytes, {
        executionProviders: ["webgpu", "wasm"]
      });
    } catch (e2) {
      throw new Error(
        (e1 && e1.message ? e1.message : String(e1)) +
          " | webgpu fallback: " +
          (e2 && e2.message ? e2.message : String(e2))
      );
    }
  }
}

self.onmessage = async function (e) {
  var type = e.data.type;

  if (type === "init") {
    var modelBytes = e.data.modelBytes;
    config = e.data.config;
    try {
      session = await createSession(modelBytes);
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({
        type: "error",
        message: err && err.message ? err.message : String(err)
      });
    }
    return;
  }

  if (type === "speak") {
    generating = true;
    var segments = e.data.segments;
    var noise_scale = e.data.noise_scale;
    var noise_w = e.data.noise_w;

    try {
      var items = Array.isArray(segments)
        ? segments
        : [{ text: e.data.text, lengthScale: e.data.lengthScale || 1.0 }];
      var total = items.length;

      for (var i = 0; i < total; i++) {
        if (!generating) break;

        var item = items[i] || {};
        var text = String(item.text || "").trim();
        if (!text) continue;

        var sentences = await self.phonemize(text, config.espeak.voice);
        var idMap = config.phoneme_id_map;
        var lengthScale = Number(item.lengthScale) || 1.0;

        for (var si = 0; si < sentences.length; si++) {
          if (!generating) break;
          var ipa = sentences[si];
          if (!ipa || !ipa.trim()) continue;

          var ids = phonemesToIds(ipa, idMap);
          if (ids.length <= 2) continue;

          var feeds = {
            input: new self.ort.Tensor(
              "int64",
              BigInt64Array.from(ids.map(BigInt)),
              [1, ids.length]
            ),
            input_lengths: new self.ort.Tensor(
              "int64",
              BigInt64Array.from([BigInt(ids.length)]),
              [1]
            ),
            scales: new self.ort.Tensor(
              "float32",
              Float32Array.from([
                Number(noise_scale) || 0.667,
                lengthScale,
                Number(noise_w) || 0.8
              ]),
              [3]
            )
          };
          var out = await session.run(feeds);
          var audio = out.output.data;
          self.postMessage(
            { type: "chunk", audio: audio, utterance: i + 1, total: total },
            [audio.buffer]
          );
        }
      }

      self.postMessage({ type: "done" });
    } catch (err) {
      self.postMessage({
        type: "error",
        message: err && err.message ? err.message : String(err)
      });
    }
    generating = false;
    return;
  }

  if (type === "stop") {
    generating = false;
  }
};
