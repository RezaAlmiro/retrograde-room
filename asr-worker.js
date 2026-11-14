// Retrograde Room v6: the transcript worker. Libraries come from jsDelivr by import (exact versions); every data
// file (models, runtime binary, Quran text) is published with the artifact and fetched by relative URL, checked
// against its SHA-256, and kept in Cache Storage so it downloads once per browser.
import * as Q from './asr-core.js';

const TJS = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js';
const ORT_GLUE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/ort-wasm-simd-threaded.mjs';
const BASE = new URL('./', self.location.href);
const CACHE = 'rr-asr-v1', VPATH = '/rr-asr/';
const SR = 16000;

let manifestP = null, tjsP = null, ctcP = null, whisperP = null, quranP = null;
let current = 0;                     // id of the job that may run; older jobs stop at their next checkpoint
const post = (m) => self.postMessage(m);
class Stop extends Error {}
const check = (id) => { if (id !== current) throw new Stop(); };

// ---------- files ----------
const manifest = () => (manifestP ||= fetch(new URL('asr-manifest.json', BASE)).then((r) => {
  if (!r.ok) throw new Error('asr-manifest.json: ' + r.status);
  return r.json();
}));
const openCache = async () => { try { return await caches.open(CACHE); } catch { return null; } };
const keyOf = (m) => 'https://rr-asr.invalid/' + m.sha256;
async function cached(names) {
  const man = await manifest(), c = await openCache();
  let need = 0;
  for (const n of names) { const m = man.files[n]; if (!c || !(await c.match(keyOf(m)))) need += m.size; }
  return need;
}
function b64(text) {
  if (Uint8Array.fromBase64) return Uint8Array.fromBase64(text);
  const bin = atob(text), out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
const concat = (parts, n) => { const out = new Uint8Array(n); let o = 0; for (const p of parts) { out.set(p, o); o += p.length; } return out; };
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
const loaded = new Map();            // name -> Promise<Uint8Array>
function file(name, onBytes = () => {}) {
  if (loaded.has(name)) return loaded.get(name);
  const p = (async () => {
    const m = (await manifest()).files[name];
    if (!m) throw new Error('not published: ' + name);
    const c = await openCache();
    if (c) { const hit = await c.match(keyOf(m)); if (hit) return new Uint8Array(await hit.arrayBuffer()); }
    const parts = [];
    for (const src of m.src) {
      const r = await fetch(new URL(src, BASE));
      if (!r.ok) throw new Error(src + ': ' + r.status);
      const chunks = []; let n = 0;
      const rd = r.body.getReader();
      for (;;) { const { done, value } = await rd.read(); if (done) break; chunks.push(value); n += value.length; onBytes(m.b64 ? value.length * 0.75 : value.length); }
      const raw = concat(chunks, n);
      parts.push(m.b64 ? b64(new TextDecoder().decode(raw)) : raw);
    }
    const bytes = concat(parts, m.size);
    if (bytes.length !== m.size || hex(await crypto.subtle.digest('SHA-256', bytes)) !== m.sha256) throw new Error(name + ' arrived damaged; try again');
    if (c) { try { await c.put(keyOf(m), new Response(bytes)); } catch { /* storage full or blocked: works without */ } }
    return bytes;
  })();
  loaded.set(name, p);
  p.catch(() => loaded.delete(name));
  return p;
}

// ---------- runtime and models ----------
function tjs(onBytes) {
  return (tjsP ||= (async () => {
    const T = await import(TJS);
    const wasm = await file('ort.wasm', onBytes);
    const ox = T.env.backends.onnx;                        // its .wasm object is the runtime's own settings
    ox.wasm.numThreads = 1;                                // no SharedArrayBuffer here: one thread
    ox.wasm.proxy = false;
    ox.wasm.wasmPaths = { mjs: ORT_GLUE, wasm: new URL('asr-ort.wasm', BASE).href };
    ox.wasm.wasmBinary = wasm;                             // the CPU build (14 MB) published with the page
    T.env.useWasmCache = false;
    T.env.useBrowserCache = false;
    T.env.allowRemoteModels = false;
    T.env.allowLocalModels = true;
    T.env.localModelPath = VPATH;
    const net = T.env.fetch || fetch;
    T.env.fetch = async (url, init) => {
      const u = String(url);
      if (!u.startsWith(VPATH)) return net(url, init);
      const name = u.slice(VPATH.length);
      if (name === 'ctc/config.json') return new Response('{"model_type":"wav2vec2","architectures":["Wav2Vec2Model"]}', { headers: { 'content-type': 'application/json' } });
      const real = name === 'ctc/onnx/model.onnx' ? 'ctc.onnx' : name;
      const man = await manifest();
      if (!man.files[real]) return new Response(null, { status: 404 });
      const bytes = await file(real);
      return new Response(bytes, { headers: { 'content-length': String(bytes.length) } });
    };
    return T;
  })());
}
function ctcModel(onBytes) {
  return (ctcP ||= (async () => {
    const T = await tjs(onBytes);
    await file('ctc.onnx', onBytes);
    return T.Wav2Vec2Model.from_pretrained('ctc', { dtype: 'fp32', device: 'wasm' });
  })());
}
function quranText(onBytes) {
  return (quranP ||= file('quran.json', onBytes).then((b) => Q.buildQuran(JSON.parse(new TextDecoder().decode(b)))));
}
function whisper(onBytes) {
  return (whisperP ||= (async () => {
    const T = await tjs(onBytes);
    for (const n of ['whisper/onnx/encoder_model_quantized.onnx', 'whisper/onnx/decoder_model_merged_quantized.onnx', 'whisper/tokenizer.json']) await file(n, onBytes);
    return T.pipeline('automatic-speech-recognition', 'whisper', { dtype: 'q8', device: 'wasm' });
  })());
}
const NEEDS = {
  base: ['ort.wasm'],
  quran: ['ctc.onnx', 'quran.json'],
  whisper: ['whisper/onnx/encoder_model_quantized.onnx', 'whisper/onnx/decoder_model_merged_quantized.onnx', 'whisper/tokenizer.json',
    'whisper/tokenizer_config.json', 'whisper/config.json', 'whisper/generation_config.json', 'whisper/preprocessor_config.json'],
};

// ---------- Quran ----------
async function runCtc(model, T, pcm, a, b) {
  const F = Q.fbank(pcm, a, b);
  if (F.T < 8) return { lp: new Float32Array(0), Tout: 0 };
  const out = await model.forward({
    x: new T.Tensor('float32', F.data, [1, F.T, 80]),
    x_lens: new T.Tensor('int64', BigInt64Array.from([BigInt(F.T)]), [1]),
  });
  const Tout = Number(out.log_probs_len.data[0]);
  return { lp: out.log_probs.data.slice(0, Tout * Q.VOCAB.length), Tout };
}
// Quran display: the ayahs heard, whole, in Tanzil's text; the isti'adha only when recited. Returns the common
// transcript shape {kind, dir, lang, label, blocks, toks, occ}.
function quranView(tr, QD) {
  const W = QD.W, toks = [], blocks = [], occ = [];
  if (!tr.words.length) return null;
  const tokOf = new Map();
  const heard = new Set(tr.words.map((x) => x.j));
  const order = [];                                   // suras in the order they are first heard
  for (const x of tr.words) { const s = W[x.j].s; if (!order.includes(s)) order.push(s); }
  const ranges = [];
  for (const s of order) {
    const as = tr.words.filter((x) => W[x.j].s === s && W[x.j].a > 0).map((x) => W[x.j].a);
    const ist = tr.words.some((x) => W[x.j].s === s && W[x.j].opt === 2);
    const lo = as.length ? Math.min(...as) : 1, hi = as.length ? Math.max(...as) : 1;
    ranges.push([s, lo, hi]);
    const sura = QD.data.suras[s - 1];
    blocks.push({ type: 'sura', s, ar: sura.ar, tr: sura.tr, en: sura.en });
    const add = (j) => { tokOf.set(j, toks.length); toks.push(W[j].text); return toks.length - 1; };
    if (ist) { const ids = []; for (let j = QD.suraStart[s]; W[j].opt === 2; j++) ids.push(add(j)); blocks.push({ type: 'ist', toks: ids }); }
    for (let a = lo; a <= hi; a++) {
      const j0 = QD.ayahStart[s][a], j1 = a < sura.ayahs.length ? QD.ayahStart[s][a + 1] : (s < 114 ? QD.suraStart[s + 1] : W.length);
      const bas = [], ids = [];
      for (let j = j0; j < j1; j++) (W[j].opt === 1 ? bas : ids).push(add(j));
      if (bas.length) blocks.push({ type: 'bas', toks: bas });
      blocks.push({ type: 'ayah', s, a, toks: ids });
    }
  }
  for (const x of tr.words) if (tokOf.has(x.j)) occ.push({ w: tokOf.get(x.j), t0: +x.t0.toFixed(3), t1: +x.t1.toFixed(3), r: x.rep ? 1 : 0 });
  occ.sort((x, y) => x.t0 - y.t0);
  const label = ranges.map(([s, lo, hi]) => `${QD.data.suras[s - 1].tr} ${lo === hi ? lo : lo + '–' + hi}`).join(', ');
  return { kind: 'quran', dir: 'rtl', lang: 'ar', label, blocks, toks, occ, heard: tr.words.length };
}

// ---------- speech (Whisper) ----------
async function detectLanguage(asr, T, pcm) {
  const n = Math.min(pcm.length, 30 * SR);
  const { input_features } = await asr.processor(pcm.subarray(0, n));
  const gc = asr.model.generation_config;
  const sot = gc.decoder_start_token_id;
  const out = await asr.model({ input_features, decoder_input_ids: new T.Tensor('int64', BigInt64Array.from([BigInt(sot)]), [1, 1]) });
  const logits = out.logits.data, V = out.logits.dims.at(-1), off = logits.length - V;
  let best = null, bv = -Infinity;
  for (const [tok, id] of Object.entries(gc.lang_to_id)) if (logits[off + id] > bv) { bv = logits[off + id]; best = tok.slice(2, -2); }
  return best || 'en';
}
function speechView(words, lang) {
  const rtl = ['ar', 'fa', 'ur', 'he', 'yi', 'ps', 'sd'].includes(lang);
  const toks = [], occ = [], blocks = [];
  let para = null, lastEnd = -1;
  for (const w of words) {
    if (!para || w.t0 - lastEnd > 1.6) { para = { type: 'para', toks: [] }; blocks.push(para); }
    occ.push({ w: toks.length, t0: +w.t0.toFixed(3), t1: +w.t1.toFixed(3) });
    para.toks.push(toks.length); toks.push(w.text);
    lastEnd = w.t1;
  }
  return { kind: 'speech', dir: rtl ? 'rtl' : 'ltr', lang, label: '', blocks, toks, occ };
}

// ---------- a job ----------
async function transcribe({ id, pcm, lang }) {
  const status = (phase, extra = {}) => post({ type: 'status', id, phase, ...extra });
  let dl = { done: 0, total: 0, last: 0 };
  const onBytes = (n) => {
    dl.done += n;
    const now = Date.now();
    if (now - dl.last > 250) { dl.last = now; status('download', { done: Math.min(dl.done, dl.total), total: dl.total }); }
  };
  const want = async (names) => { dl.total = await cached([...NEEDS.base, ...names]); dl.done = 0; if (dl.total) status('download', { done: 0, total: dl.total }); };
  const dur = pcm.length / SR;

  let quranTr = null;
  if (lang !== 'en') {
    await want(NEEDS.quran);
    status('load', { what: 'quran' });
    const [T, model, QD] = await Promise.all([tjs(onBytes), ctcModel(onBytes), quranText(onBytes)]);
    check(id);
    const ranges = Q.splitAtPauses(pcm);
    const pieces = [];
    let lastPost = 0, notQuran = false;
    for (let k = 0; k < ranges.length; k++) {
      const r = ranges[k];
      const { lp, Tout } = await runCtc(model, T, pcm, r.a, r.b);
      check(id);
      pieces.push({ ...r, lp, Tout, words: Q.greedyWords(lp, Tout) });
      status('listen', { done: r.b / SR, total: dur, what: 'quran' });
      // after about 40 s of audio decide whether this is the Quran at all (Auto)
      const heardS = r.b / SR;
      if (lang === 'auto' && !quranTr && heardS > Math.min(40, dur * 0.9)) {
        const probe = Q.transcribeQuran(pieces, QD);
        if (probe.total < 4 || probe.matched < Math.max(3, probe.total * 0.45)) { notQuran = true; break; }
        quranTr = probe;
      }
      if (heardS - lastPost > 20 && k < ranges.length - 1) {
        lastPost = heardS;
        const tr = Q.transcribeQuran(pieces, QD);
        if (tr.matched >= Math.max(3, tr.total * 0.45)) {
          quranTr = tr;
          const view = quranView(tr, QD);
          if (view) post({ type: 'result', id, final: k === ranges.length - 1, view });
        }
      }
    }
    if (!notQuran) {
      const tr = Q.transcribeQuran(pieces, QD), view = tr.matched >= Math.max(3, tr.total * 0.45) && quranView(tr, QD);
      if (view) { post({ type: 'result', id, final: true, view }); return; }
    }
  }

  // speech: Whisper, in 30 s windows that overlap by 5 s (words shown as each window finishes)
  await want(NEEDS.whisper);
  status('load', { what: 'speech' });
  const T = await tjs(onBytes);
  const asr = await whisper(onBytes);
  check(id);
  let code = lang;
  if (lang === 'auto' || lang === 'ar') {
    if (lang === 'auto') { status('listen', { done: 0, total: dur, what: 'language' }); code = await detectLanguage(asr, T, pcm).catch(() => 'en'); }
    check(id);
  }
  const WIN = 30, HOP = 25, words = [];
  for (let s = 0; s < dur; s += HOP) {
    const a = Math.floor(s * SR), b = Math.min(pcm.length, Math.floor((s + WIN) * SR));
    const out = await asr(pcm.subarray(a, b), { language: code, task: 'transcribe', return_timestamps: 'word' });
    check(id);
    const keepFrom = s === 0 ? -1 : s + 2.5, keepTo = b >= pcm.length ? Infinity : s + HOP + 2.5;
    for (const c of out.chunks || []) {
      const text = String(c.text).trim();
      if (!text) continue;
      const t0 = s + (c.timestamp[0] ?? 0), t1 = s + (c.timestamp[1] ?? (c.timestamp[0] ?? 0) + 0.3);
      if (t0 < keepFrom || t0 >= keepTo) continue;
      const prev = words[words.length - 1];
      if (prev && Math.abs(prev.t0 - t0) < 0.3 && prev.text === text) continue;
      words.push({ text, t0, t1: Math.max(t1, t0 + 0.05) });
    }
    const last = b >= pcm.length;
    status('listen', { done: Math.min(dur, s + WIN), total: dur, what: 'speech', lang: code });
    post({ type: 'result', id, final: last, view: { ...speechView(words, code), label: code } });
    if (last) break;
  }
}

self.addEventListener('securitypolicyviolation', (e) => post({ type: 'csp', directive: e.violatedDirective, uri: String(e.blockedURI).slice(0, 120) }));
// One job at a time: a new job (or a cancel) stops the running one at its next checkpoint, and the new one starts only
// after it has stopped, so two jobs never run the same model at once.
let chain = Promise.resolve();
async function runJob(m) {
  if (m.id !== current) return;                      // replaced while it waited
  try { await transcribe(m); }
  catch (err) {
    if (err instanceof Stop) return;
    post({ type: 'error', id: m.id, message: String(err && err.message || err) });
  }
}
self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'cancel') { current = 0; return; }
  if (m.type !== 'transcribe') return;
  current = m.id;
  chain = chain.then(() => runJob(m));
};
post({ type: 'ready' });
