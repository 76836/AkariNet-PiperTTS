/**
 * Akari TTS — main application logic (standalone demo).
 * For AkariNet integration see the characters/akari/PiperTTS.js loader
 * in the main Akari repo, which points at this Pages site.
 *
 * Upload model.onnx, ort-wasm.wasm, phonemizer.js, ort.all.min.js first.
 */

const INFERENCE = { noise_scale: 0.667, noise_w: 0.8 };

const $text = document.getElementById("text");
const $speak = document.getElementById("speak");
const $stop = document.getElementById("stop");
const $status = document.getElementById("status");
const $audio = document.getElementById("audio");
const $speed = document.getElementById("speed");
const $speedVal = document.getElementById("speedVal");

const IS_FILE = location.protocol === "file:";

let config = null;
let session = null;
let worker = null;
let ready = false;

function loadScript(src) {
  return new Promise((resolve, reject) => {
    const s = document.createElement("script");
    s.src = src;
    s.onload = resolve;
    s.onerror = () => reject(new Error("Failed to load script: " + src));
    document.head.appendChild(s);
  });
}

async function b64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (bytes.length >= 2 && bytes[0] === 0x1f && bytes[1] === 0x8b) {
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  return bytes;
}

async function loadBytes(url, fallback, varName) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return new Uint8Array(await resp.arrayBuffer());
  } catch (e) {
    if (fallback) {
      await loadScript(fallback);
      return await b64ToBytes(window[varName]);
    }
    throw e;
  }
}

async function loadBlobUrl(url, fallback, varName, mime) {
  const bytes = await loadBytes(url, fallback, varName);
  return URL.createObjectURL(new Blob([bytes], { type: mime }));
}

async function loadConfig(url, fallback) {
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error("HTTP " + resp.status);
    return await resp.json();
  } catch (e) {
    if (fallback) {
      await loadScript(fallback);
      return window.__modelConfig;
    }
    throw e;
  }
}

class AudioQueue {
  constructor(audioEl) {
    this.el = audioEl;
    this.queue = [];
    this.playing = false;
    this.el.addEventListener("ended", () => this._next());
  }
  add(blob) {
    this.queue.push(blob);
    if (!this.playing) this._next();
  }
  _next() {
    if (this.queue.length === 0) { this.playing = false; return; }
    this.playing = true;
    this.el.src = URL.createObjectURL(this.queue.shift());
    this.el.play();
  }
  clear() {
    this.queue = [];
    this.playing = false;
    this.el.pause();
    this.el.removeAttribute("src");
  }
}

const audioQueue = new AudioQueue($audio);

function phonemesToIds(ipa, idMap) {
  const ids = [];
  ids.push(...idMap["^"], ...idMap["_"]);
  for (const ch of ipa.trim().normalize("NFD")) {
    if (ch in idMap) ids.push(...idMap[ch], ...idMap["_"]);
  }
  ids.push(...idMap["$"]);
  return ids;
}

function encodeWav(samples, sampleRate) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) v.setUint8(o + i, s.charCodeAt(i)); };
  str(0, "RIFF");     v.setUint32(4, 36 + samples.length * 2, true);
  str(8, "WAVE");     str(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true);   v.setUint16(34, 16, true);
  str(36, "data");    v.setUint32(40, samples.length * 2, true);
  for (let i = 0, o = 44; i < samples.length; i++, o += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    v.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return buf;
}

async function init() {
  config = await loadConfig("config.json", "fallback/config.js");

  if (IS_FILE) {
    await initMainThread();
  } else {
    try {
      await initWorker();
    } catch (e) {
      console.warn("Worker init failed, falling back to main thread:", e);
      await initMainThread();
    }
  }
}

async function initMainThread() {
  setStatus("Loading ORT runtime…");
  const mjsUrl = await loadBlobUrl("ort-wasm.mjs", "fallback/ort-mjs.b64.js", "__ortMjsB64", "text/javascript");
  const wasmUrl = await loadBlobUrl("ort-wasm.wasm", "fallback/ort-wasm.b64.js", "__ortWasmB64", "application/wasm");

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = { mjs: mjsUrl, wasm: wasmUrl };

  setStatus("Loading voice model…");
  const modelBytes = await loadBytes("https://huggingface.co/76836-HW/AkariNet-PiperTTS/resolve/main/model.onnx", "fallback/model.b64.js", "__modelB64");
  session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: [{ name: "wasm", simd: true }],
  });

  ready = true;
  setStatus("Ready" + (IS_FILE ? " (main thread)" : " (main thread fallback)"));
  $speak.disabled = false;
}

async function initWorker() {
  setStatus("Starting worker…");
  worker = new Worker("worker.js");

  setStatus("Loading voice model…");
  const modelBytes = await loadBytes("https://huggingface.co/76836-HW/AkariNet-PiperTTS/resolve/main/model.onnx", "fallback/model.b64.js", "__modelB64");

  await new Promise((resolve, reject) => {
    const handler = (e) => {
      if (e.data.type === "ready") { worker.removeEventListener("message", handler); resolve(); }
      else if (e.data.type === "error") { worker.removeEventListener("message", handler); reject(new Error(e.data.message)); }
    };
    worker.addEventListener("message", handler);
    worker.postMessage({ type: "init", modelBytes: modelBytes.buffer, config }, [modelBytes.buffer]);
  });

  worker.onmessage = (e) => {
    const { type, audio, sentence, total, error } = e.data;
    if (type === "chunk") {
      const wav = encodeWav(new Float32Array(audio), config.audio.sample_rate);
      audioQueue.add(new Blob([wav], { type: "audio/wav" }));
      setStatus(`Generating… ${sentence}/${total} sentences`);
    } else if (type === "done") {
      setStatus("Done");
      $speak.disabled = false;
    } else if (type === "error") {
      setStatus("Error: " + error);
      $speak.disabled = false;
    }
  };

  ready = true;
  setStatus("Ready (worker)");
  $speak.disabled = false;
}

async function speak() {
  const text = $text.value.trim();
  if (!text || !ready) return;

  audioQueue.clear();
  $speak.disabled = true;
  $stop.disabled = false;
  setStatus("Generating…");

  const lengthScale = 1.0 / parseFloat($speed.value);

  if (worker) {
    worker.postMessage({ type: "speak", text, lengthScale, ...INFERENCE });
  } else {
    await generateMainThread(text, lengthScale);
  }
}

async function generateMainThread(text, lengthScale) {
  const sentences = await phonemize(text, config.espeak.voice);
  const idMap = config.phoneme_id_map;
  const sr = config.audio.sample_rate;
  const { noise_scale, noise_w } = INFERENCE;
  const total = sentences.length;

  for (let i = 0; i < total; i++) {
    const ipa = sentences[i];
    if (!ipa || !ipa.trim()) continue;
    const ids = phonemesToIds(ipa, idMap);
    if (ids.length <= 2) continue;

    setStatus(`Generating… ${i + 1}/${total} sentences`);

    const feeds = {
      input: new ort.Tensor("int64", BigInt64Array.from(ids.map(BigInt)), [1, ids.length]),
      input_lengths: new ort.Tensor("int64", BigInt64Array.from([BigInt(ids.length)]), [1]),
      scales: new ort.Tensor("float32", Float32Array.from([noise_scale, lengthScale, noise_w]), [3]),
    };
    const out = await session.run(feeds);
    const wav = encodeWav(new Float32Array(out.output.data), sr);
    audioQueue.add(new Blob([wav], { type: "audio/wav" }));

    await new Promise(r => setTimeout(r, 0));
  }

  setStatus("Done");
  $speak.disabled = false;
  $stop.disabled = true;
}

function stop() {
  audioQueue.clear();
  if (worker) worker.postMessage({ type: "stop" });
  $speak.disabled = false;
  $stop.disabled = true;
  setStatus("Stopped");
}

function setStatus(msg) { $status.textContent = msg; }

$speak.addEventListener("click", speak);
$stop.addEventListener("click", stop);
$speed.addEventListener("input", () => { $speedVal.textContent = parseFloat($speed.value).toFixed(1) + "x"; });

init().catch((e) => { console.error(e); setStatus("Failed to load: " + e.message); });
