// Retrograde Room v6 — Quran recitation to Tanzil words with times. Plain JS, no dependencies.
// Model: Abdou/quran-asr-small (CTC, 31 tokens: blank, 29 letters, word separator; 40 ms frames).
// Steps: pauses split the audio (the model was trained on single ayahs) -> fbank -> model per piece
// -> greedy letters -> words -> located in the Quran (alignment with jumps for repeats)
// -> CTC forced alignment of the canonical words in each piece -> word times.

export const LETTERS = 'ءابتثجحخدذرزسشصضطظعغفقكلمنهوي';
export const VOCAB = ['<blk>', ...LETTERS, '▁'];
export const SEP = 30;
const TOK = new Map(VOCAB.map((t, i) => [t, i]));
const FOLD = { 'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا', 'ٲ': 'ا', 'ٳ': 'ا', 'ى': 'ي', 'ئ': 'ي', 'ے': 'ي', 'ی': 'ي', 'ؤ': 'و', 'ة': 'ه', 'ک': 'ك' };
const DIGIT = /\p{Nd}/u;
function isMark(c) {
  const o = c.codePointAt(0);
  return (o >= 0x64B && o <= 0x65F) || o === 0x670 || (o >= 0x6D6 && o <= 0x6ED) || o === 0x640 || o === 0x200F || DIGIT.test(c);
}
// Port of quran_asr/text.py normalize_word: letters only, alif forms folded.
export function normWord(w) {
  let s = '';
  for (const ch of w) {
    if (isMark(ch)) continue;
    const c = FOLD[ch] || ch;
    if (c.length === 1 && TOK.has(c) && c !== '▁') s += c;
  }
  return s;
}

// ---------- features: port of quran_asr/features.py (Kaldi fbank, 80 bins, snip_edges false) ----------
const SR = 16000, SHIFT = 160, LEN = 400, NFFT = 512, NB = 80, EPS = 1.1920928955078125e-7;
const mel = (f) => 1127 * Math.log(1 + f / 700);
let BANKS = null, WIN = null, FFTW = null;
function initFeat() {
  if (BANKS) return;
  const bins = NFFT / 2, width = SR / NFFT, lo = mel(20), hi = mel(SR / 2 - 400), d = (hi - lo) / (NB + 1);
  BANKS = [];
  for (let b = 0; b < NB; b++) {
    const l = lo + b * d, c = lo + (b + 1) * d, r = lo + (b + 2) * d, ws = [];
    let first = -1;
    for (let i = 0; i < bins; i++) {
      const m = mel(width * i);
      if (l < m && m < r) { if (first < 0) first = i; ws.push(m <= c ? (m - l) / (c - l) : (r - m) / (r - c)); }
    }
    BANKS.push({ lo: first, w: Float32Array.from(ws) });
  }
  WIN = new Float32Array(LEN);
  for (let i = 0; i < LEN; i++) WIN[i] = Math.pow(0.5 - 0.5 * Math.cos(2 * Math.PI * i / (LEN - 1)), 0.85);
  const cos = new Float64Array(NFFT / 2), sin = new Float64Array(NFFT / 2), rev = new Uint16Array(NFFT);
  for (let i = 0; i < NFFT / 2; i++) { cos[i] = Math.cos(2 * Math.PI * i / NFFT); sin[i] = -Math.sin(2 * Math.PI * i / NFFT); }
  for (let i = 0, bits = Math.log2(NFFT); i < NFFT; i++) { let r = 0; for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b); rev[i] = r; }
  FFTW = { cos, sin, rev };
}
function fft(re, im) {
  const { cos, sin, rev } = FFTW, n = NFFT;
  for (let i = 0; i < n; i++) { const j = rev[i]; if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; } }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1, step = n / size;
    for (let s = 0; s < n; s += size) {
      for (let k = 0; k < half; k++) {
        const wr = cos[k * step], wi = sin[k * step], a = s + k, b = a + half;
        const tr = re[b] * wr - im[b] * wi, ti = re[b] * wi + im[b] * wr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
      }
    }
  }
}
// wave: Float32Array, 16 kHz mono, [-1, 1]. Returns { T, data: Float32Array(T*80) }.
export function fbank(wave, a = 0, b = wave.length) {
  initFeat();
  const n = b - a, T = Math.floor((n + SHIFT / 2) / SHIFT), out = new Float32Array(Math.max(0, T) * NB);
  const fr = new Float32Array(LEN), re = new Float64Array(NFFT), im = new Float64Array(NFFT), pw = new Float64Array(NFFT / 2);
  for (let t = 0; t < T; t++) {
    let mean = 0;
    for (let i = 0; i < LEN; i++) {
      let k = t * SHIFT + SHIFT / 2 - LEN / 2 + i;
      if (k < 0) k = -k - 1;
      if (k >= n) k = 2 * n - 1 - k;
      if (k < 0) k = 0; else if (k > n - 1) k = n - 1;
      fr[i] = wave[a + k]; mean += fr[i];
    }
    mean /= LEN;
    for (let i = 0; i < LEN; i++) fr[i] -= mean;
    re.fill(0); im.fill(0);
    re[0] = fr[0] * (1 - 0.97) * WIN[0];
    for (let i = 1; i < LEN; i++) re[i] = (fr[i] - 0.97 * fr[i - 1]) * WIN[i];
    fft(re, im);
    for (let i = 0; i < NFFT / 2; i++) pw[i] = re[i] * re[i] + im[i] * im[i];
    for (let m = 0; m < NB; m++) {
      const { lo, w } = BANKS[m];
      let e = 0;
      for (let i = 0; i < w.length; i++) e += pw[lo + i] * w[i];
      out[t * NB + m] = Math.log(Math.max(e, EPS));
    }
  }
  return { T, data: out };
}

// ---------- pauses: split the recording into utterances (the model was trained on single ayahs) ----------
// Returns [{a, b}] sample ranges (with a little silence kept around the speech), none longer than maxS.
export function splitAtPauses(w, { minPause = 0.22, maxS = 16, pad = 0.12 } = {}) {
  const hop = 160, n = Math.floor(w.length / hop);
  if (n < 10) return n ? [{ a: 0, b: w.length }] : [];
  const db = new Float32Array(n);
  for (let k = 0; k < n; k++) { let e = 0; for (let i = k * hop; i < (k + 1) * hop; i++) e += w[i] * w[i]; db[k] = 10 * Math.log10(e / hop + 1e-12); }
  const sm = new Float32Array(n);                     // 50 ms moving average
  for (let k = 0; k < n; k++) { let s = 0, c = 0; for (let j = Math.max(0, k - 2); j <= Math.min(n - 1, k + 2); j++) { s += db[j]; c++; } sm[k] = s / c; }
  const sorted = Float32Array.from(sm).sort();
  const floor = sorted[Math.floor(n * 0.08)], peak = sorted[Math.floor(n * 0.97)];
  const thr = floor + Math.max(5, 0.28 * (peak - floor));
  const speech = new Uint8Array(n);
  for (let k = 0; k < n; k++) speech[k] = sm[k] > thr ? 1 : 0;
  // runs of speech separated by pauses of at least minPause
  const runs = []; let k = 0;
  const minGap = Math.round(minPause * 100);
  while (k < n) {
    while (k < n && !speech[k]) k++;
    if (k >= n) break;
    let s = k, e = k;
    for (;;) {
      while (e < n && speech[e]) e++;
      let g = e; while (g < n && !speech[g]) g++;
      if (g < n && g - e < minGap) { e = g; continue; }
      break;
    }
    runs.push([s, e]); k = e;
  }
  // drop clicks (< 80 ms) and keep a little silence around each run
  const P = Math.round(pad * 100), segs = [];
  for (const [s, e] of runs) { if (e - s < 8) continue; segs.push([Math.max(0, s - P), Math.min(n, e + P)]); }
  for (let i = 1; i < segs.length; i++) if (segs[i][0] < segs[i - 1][1]) { const m = (segs[i][0] + segs[i - 1][1]) >> 1; segs[i - 1][1] = m; segs[i][0] = m; }
  // split long runs at the quietest 120 ms in their middle
  const maxF = Math.round(maxS * 100), out = [];
  const cut = (s, e) => {
    if (e - s <= maxF) { out.push([s, e]); return; }
    let best = -1, bv = Infinity;
    for (let j = s + Math.round((e - s) * 0.3); j < s + Math.round((e - s) * 0.7); j++) {
      let v = 0; for (let q = j - 6; q <= j + 6; q++) v += sm[q]; if (v < bv) { bv = v; best = j; }
    }
    cut(s, best); cut(best, e);
  };
  for (const [s, e] of segs) cut(s, e);
  return out.map(([s, e]) => ({ a: s * hop, b: Math.min(w.length, e * hop) }));
}

// ---------- greedy CTC ----------
// lp: Float32Array(Tout*31) log-probs of one piece. Returns words [{w, f0, f1, conf}] (frames local to the piece).
export function greedyWords(lp, Tout) {
  const V = VOCAB.length, words = [];
  let prev = -1, cur = null;
  for (let t = 0; t < Tout; t++) {
    let a = 0, best = lp[t * V];
    for (let v = 1; v < V; v++) if (lp[t * V + v] > best) { best = lp[t * V + v]; a = v; }
    if (a !== prev && a !== 0) {
      if (a === SEP) { if (cur) { words.push(cur); cur = null; } }
      else { if (!cur) cur = { w: '', f0: t, f1: t, conf: 0, n: 0 }; cur.w += VOCAB[a]; cur.f1 = t; cur.conf += best; cur.n++; }
    } else if (a === prev && a !== 0 && a !== SEP && cur) cur.f1 = t;
    prev = a;
  }
  if (cur) words.push(cur);
  for (const x of words) { x.conf = Math.exp(x.conf / x.n); delete x.n; }
  return words;
}

// ---------- the Quran as a word list ----------
// data: {suras:[{n, ar, tr, en, ayahs:[text]}]} (Tanzil Uthmani). Display words keep their marks; tokens that are only
// marks (pause signs, sajdah) join the word before. The basmala that opens ayah 1 of suras 2-114 (not 9) and the
// isti'adha (composed from Tanzil's own words: 2:67 and 16:98) are optional: skipping them costs nothing.
export function buildQuran(data) {
  const W = [];          // {s, a, k, text, norm, opt}  opt: 0 real, 1 basmala, 2 isti'adha
  const BAS = ['بسم', 'الله', 'الرحمن', 'الرحيم'];
  const ayahTokens = (s, a) => data.suras[s - 1].ayahs[a - 1].split(' ');
  const words = (s, a) => {
    const out = [];
    for (const t of ayahTokens(s, a)) {
      const n = normWord(t);
      if (!n) { if (out.length) out[out.length - 1].text += ' ' + t; else out.push({ text: t, norm: '' }); }
      else if (out.length && !out[out.length - 1].norm) { out[out.length - 1].text += ' ' + t; out[out.length - 1].norm = n; }
      else out.push({ text: t, norm: n });
    }
    return out;
  };
  const find = (s, a, seq) => {
    const ws = words(s, a);
    for (let i = 0; i + seq.length <= ws.length; i++) if (seq.every((q, j) => ws[i + j].norm === q)) return ws.slice(i, i + seq.length);
    return null;
  };
  const ist = [...(find(2, 67, ['اعوذ']) || []), ...(find(16, 98, ['بالله', 'من', 'الشيطان', 'الرجيم']) || [])];
  const suraStart = [], ayahStart = [];
  for (const sura of data.suras) {
    const s = sura.n;
    suraStart[s] = W.length;
    if (ist.length === 5) ist.forEach((x, k) => W.push({ s, a: 0, k, text: x.text.replace(/[ ]*[ۖ-ۜ۩]+$/u, ''), norm: x.norm, opt: 2 }));
    ayahStart[s] = [];
    for (let a = 1; a <= sura.ayahs.length; a++) {
      ayahStart[s][a] = W.length;
      const ws = words(s, a);
      const bas = s !== 1 && s !== 9 && a === 1 && ws.length > 4 && BAS.every((q, j) => ws[j].norm === q);
      ws.forEach((x, k) => W.push({ s, a, k, text: x.text, norm: x.norm, opt: bas && k < 4 ? 1 : 0 }));
    }
  }
  return { W, suraStart, ayahStart, ist: ist.length === 5, data };
}

// ---------- locate the hypothesis in the Quran ----------
function lev(a, b, cap) {
  if (a === b) return 0;
  const la = a.length, lb = b.length;
  if (Math.abs(la - lb) > cap) return cap + 1;
  let prev = new Array(lb + 1), cur = new Array(lb + 1);
  for (let j = 0; j <= lb; j++) prev[j] = j;
  for (let i = 1; i <= la; i++) {
    cur[0] = i; let rowMin = i;
    for (let j = 1; j <= lb; j++) {
      cur[j] = Math.min(cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
      if (cur[j] < rowMin) rowMin = cur[j];
    }
    if (rowMin > cap) return cap + 1;
    [prev, cur] = [cur, prev];
  }
  return prev[lb];
}
// similarity in [0,1]; below 0.5 counts as no match
export function sim(a, b) {
  if (!a || !b) return 0;
  const m = Math.max(a.length, b.length), cap = Math.floor(m / 2);
  const d = lev(a, b, cap);
  return d > cap ? 0 : 1 - d / m;
}
let BIGRAMS = null;
function bigrams(Q) {
  if (BIGRAMS && BIGRAMS.Q === Q) return BIGRAMS.map;
  const map = new Map(), W = Q.W;
  let last = -1;
  for (let j = 0; j < W.length; j++) {
    if (W[j].opt === 2) continue;                       // the isti'adha is repeated 114 times: leave it out of the votes
    if (last >= 0) { const key = W[last].norm + ' ' + W[j].norm; let v = map.get(key); if (!v) map.set(key, v = []); v.push(j); }
    last = j;
  }
  BIGRAMS = { Q, map };
  return map;
}
// Which suras are recited: votes of word pairs. Returns sorted sura numbers (may be empty).
export function suraVotes(hyp, Q) {
  const map = bigrams(Q), votes = new Map();
  for (let i = 0; i + 1 < hyp.length; i++) {
    const pos = map.get(hyp[i].w + ' ' + hyp[i + 1].w);
    if (!pos || pos.length > 60) continue;
    const per = new Map();
    for (const j of pos) per.set(Q.W[j].s, 1);
    for (const s of per.keys()) votes.set(s, (votes.get(s) || 0) + 1 / Math.sqrt(pos.length));
  }
  return votes;
}
const SC = { INS: -0.45, DEL: -0.5, JUMP: -3.5, MAXSKIP: 6, LOOK: 24 };
// hyp: [{w}] in time order. Returns {path: Int32Array (region index or -1), region: Int32Array (indices into Q.W), score, matched}
export function locate(hyp, Q, suras) {
  const region = [];
  for (const s of suras) {
    const a = Q.suraStart[s], b = s < 114 ? Q.suraStart[s + 1] : Q.W.length;
    for (let j = a; j < b; j++) region.push(j);
  }
  const R = region.length, n = hyp.length;
  const path = new Int32Array(n).fill(-1);
  if (!R || !n) return { path, region: Int32Array.from(region), score: 0, matched: 0 };
  const opt = new Uint8Array(R), norms = [];
  const cid = new Map(), cIdx = new Int32Array(R);
  region.forEach((j, r) => { opt[r] = Q.W[j].opt ? 1 : 0; const t = Q.W[j].norm; if (!cid.has(t)) { cid.set(t, norms.length); norms.push(t); } cIdx[r] = cid.get(t); });
  const hid = new Map(), hIdx = new Int32Array(n), hw = [];
  hyp.forEach((h, i) => { if (!hid.has(h.w)) { hid.set(h.w, hw.length); hw.push(h.w); } hIdx[i] = hid.get(h.w); });
  const SIM = new Float32Array(hw.length * norms.length).fill(-2);
  const score = (i, r) => {
    const q = hIdx[i] * norms.length + cIdx[r];
    let v = SIM[q];
    if (v === -2) { const s = sim(hw[hIdx[i]], norms[cIdx[r]]); v = SIM[q] = s >= 0.5 ? 2 * s - 1 : -1; }
    return v;
  };
  const back = new Uint8Array(n * R);       // 0 start, 1..LOOK diagonal from r-code, 254 insertion, 255 jump
  const jumpFrom = new Int32Array(n);
  let prev = new Float32Array(R).fill(-Infinity), cur = new Float32Array(R);
  let start = 0;                            // score of "nothing matched yet"
  for (let i = 0; i < n; i++) {
    let bp = -Infinity, bpr = -1;
    for (let r = 0; r < R; r++) if (prev[r] > bp) { bp = prev[r]; bpr = r; }
    jumpFrom[i] = bpr;
    for (let r = 0; r < R; r++) {
      const m = score(i, r);
      let best = start + m, code = 0;                                   // free start
      let skip = 0;
      for (let p = r - 1, steps = 1; p >= 0 && steps <= SC.LOOK; p--, steps++) {
        const v = prev[p] + skip * SC.DEL + m;
        if (v > best) { best = v; code = steps; }
        if (!opt[p]) skip++;
        if (skip > SC.MAXSKIP) break;
      }
      if (bpr >= 0 && bp + SC.JUMP + m > best) { best = bp + SC.JUMP + m; code = 255; }
      if (prev[r] + SC.INS > best) { best = prev[r] + SC.INS; code = 254; }
      cur[r] = best; back[i * R + r] = code;
    }
    start += SC.INS;
    [prev, cur] = [cur, prev];
  }
  let endR = -1, endV = start;
  for (let r = 0; r < R; r++) if (prev[r] > endV) { endV = prev[r]; endR = r; }
  for (let i = n - 1, r = endR; i >= 0 && r >= 0; i--) {
    const code = back[i * R + r];
    if (code === 254) continue;                       // hyp word i unmatched, state stays r
    path[i] = r;
    if (code === 0) r = -1;
    else if (code === 255) r = jumpFrom[i];
    else r = r - code;
  }
  let matched = 0;
  for (let i = 0; i < n; i++) if (path[i] >= 0 && score(i, path[i]) > 0.2) matched++;
  return { path, region: Int32Array.from(region), score: endV, matched };
}

// ---------- CTC forced alignment with a free start and end inside a piece ----------
// labels: Int array (letters and SEP). Returns {score, first: Int32Array(L), last: Int32Array(L)} frames per label.
export function ctcAlign(lp, Tout, labels) {
  const V = VOCAB.length, L = labels.length, S = 2 * L + 1;
  if (!L || !Tout) return null;
  const lab = (s) => (s & 1 ? labels[s >> 1] : 0);
  const back = new Uint8Array(Tout * S);
  let prev = new Float32Array(S).fill(-Infinity), cur = new Float32Array(S);
  let bestEnd = -Infinity, endT = -1, endS = -1;
  for (let t = 0; t < Tout; t++) {
    const row = t * V;
    for (let s = 0; s < S; s++) {
      let b = prev[s], c = 0;
      if (s > 0 && prev[s - 1] > b) { b = prev[s - 1]; c = 1; }
      if (s > 1 && (s & 1) && labels[s >> 1] !== labels[(s >> 1) - 1] && prev[s - 2] > b) { b = prev[s - 2]; c = 2; }
      if (s < 2 && 0 > b) { b = 0; c = 3; }                     // start fresh here
      cur[s] = b + lp[row + lab(s)]; back[t * S + s] = c;
    }
    for (const s of [S - 1, S - 2]) if (cur[s] > bestEnd) { bestEnd = cur[s]; endT = t; endS = s; }
    [prev, cur] = [cur, prev];
  }
  const first = new Int32Array(L).fill(-1), last = new Int32Array(L).fill(-1);
  for (let t = endT, s = endS; t >= 0; t--) {
    if (s & 1) { const l = s >> 1; if (last[l] < 0) last[l] = t; first[l] = t; }
    const c = back[t * S + s];
    if (c === 3) break;
    s -= c === 1 ? 1 : c === 2 ? 2 : 0;
  }
  return { score: bestEnd, first, last };
}
export function labelsOf(norms) {
  const out = [], wordAt = [];
  norms.forEach((w, k) => {
    if (k) { out.push(SEP); wordAt.push(-1); }
    for (const ch of w) { out.push(TOK.get(ch)); wordAt.push(k); }
  });
  return { labels: out, wordAt };
}

// ---------- the whole recitation ----------
// pieces: [{a, b, lp, Tout, words}] in time order (a, b in samples; words from greedyWords). Returns the transcript:
// {kind:'quran', suras:[n], ayahs:[{s,a,from,to}], words:[{j (index in Q.W), t0, t1}], matched, total, conf}
export const FRAME = 0.04;
export function transcribeQuran(pieces, Q, { offset = 0 } = {}) {
  const hyp = [];
  pieces.forEach((p, pi) => p.words.forEach((w) => hyp.push({ ...w, piece: pi })));
  const votes = suraVotes(hyp, Q);
  let top = 0; for (const v of votes.values()) top = Math.max(top, v);
  const suras = [...votes].filter(([, v]) => v >= Math.max(0.5, top * 0.2)).map(([s]) => s).sort((x, y) => x - y);
  const loc = locate(hyp, Q, suras);
  const { path, region } = loc;
  // canonical words per piece: matched ones plus the ones skipped between matches of the same run
  const per = pieces.map(() => []);
  let lastJ = -1, lastPiece = -1;
  for (let i = 0; i < hyp.length; i++) {
    if (path[i] < 0) continue;
    const j = region[path[i]], pi = hyp[i].piece;
    if (lastJ >= 0 && j > lastJ && j - lastJ <= SC.LOOK) {
      const gap = []; for (let q = lastJ + 1; q < j; q++) if (!Q.W[q].opt) gap.push(q);   // optional words only when heard
      if (gap.length) {
        if (pi === lastPiece) per[pi].push(...gap);
        else gap.forEach((q) => (Q.W[q].opt ? per[pi] : per[lastPiece]).push(q));   // missed basmala goes with what follows
      }
    }
    if (!(lastJ === j && lastPiece === pi)) per[pi].push(j);
    lastJ = j; lastPiece = pi;
  }
  const words = [];
  pieces.forEach((p, pi) => {
    const js = per[pi];
    if (!js.length) return;
    // split the piece's word list where it jumps (a repeat inside one piece)
    const runs = []; let run = [js[0]];
    for (let k = 1; k < js.length; k++) { if (js[k] > js[k - 1] && allOpt(Q, js[k - 1] + 1, js[k])) run.push(js[k]); else { runs.push(run); run = [js[k]]; } }
    runs.push(run);
    let f0 = 0;
    for (const r of runs) {
      const { labels, wordAt } = labelsOf(r.map((j) => Q.W[j].norm));
      const al = ctcAlign(p.lp.subarray(f0 * VOCAB.length), p.Tout - f0, labels);
      if (!al) continue;
      const t0 = new Array(r.length).fill(-1), t1 = new Array(r.length).fill(-1);
      labels.forEach((_, l) => { const k = wordAt[l]; if (k < 0 || al.first[l] < 0) return; if (t0[k] < 0) t0[k] = al.first[l]; t1[k] = al.last[l]; });
      let end = f0;
      r.forEach((j, k) => {
        if (t0[k] < 0) return;
        words.push({ j, t0: p.a / SR + (f0 + t0[k]) * FRAME + offset, t1: p.a / SR + (f0 + t1[k] + 1) * FRAME + offset });
        end = Math.max(end, f0 + t1[k] + 1);
      });
      f0 = Math.min(end, p.Tout - 1);
    }
  });
  // Pieces where nothing matched but real letters were heard: usually a repeat ("...ٱلْمَغْضُوبِ عَلَيْهِمْ" again after
  // a pause). Search their letters in the text around them, then time those words with the model.
  const span = region.length ? [region[0], region[region.length - 1]] : null;
  pieces.forEach((p, pi) => {
    if (!span || per[pi].length) return;
    const H = p.words.map((x) => x.w).join(' ');
    if (H.replace(/ /g, '').length < 5) return;
    const mid = (p.a + p.b) / 2 / SR;
    let before = -1, after = -1, tb = -Infinity, ta = Infinity;
    for (const x of words) { if (x.t0 < mid && x.t0 > tb) { tb = x.t0; before = x.j; } if (x.t0 > mid && x.t0 < ta) { ta = x.t0; after = x.j; } }
    if (before < 0 && after < 0) return;
    const c = before >= 0 ? before : after;
    const lo = Math.max(span[0], Math.min(c, after >= 0 ? after : c) - 80), hi = Math.min(span[1], Math.max(c, after >= 0 ? after : c) + 40);
    let C = ''; const owner = [];
    for (let j = lo; j <= hi; j++) { if (Q.W[j].opt === 2 || !Q.W[j].norm) continue; if (C) { C += ' '; owner.push(-1); } for (const ch of Q.W[j].norm) { C += ch; owner.push(j); } }
    const m = H.length, n = C.length;
    let prev = new Int32Array(n + 1), cur = new Int32Array(n + 1);
    const back = new Uint8Array((m + 1) * (n + 1));
    for (let i = 1; i <= m; i++) {
      cur[0] = i; back[i * (n + 1)] = 2;
      for (let j = 1; j <= n; j++) {
        let v = prev[j - 1] + (H[i - 1] === C[j - 1] ? 0 : 1), b = 0;
        if (prev[j] + 1 < v) { v = prev[j] + 1; b = 2; }
        if (cur[j - 1] + 1 < v) { v = cur[j - 1] + 1; b = 1; }
        cur[j] = v; back[i * (n + 1) + j] = b;
      }
      [prev, cur] = [cur, prev];
    }
    let bj = -1, bv = Infinity;
    for (let j = 1; j <= n; j++) {   // best end; ties go to the copy nearest the words before the pause
      const v = prev[j];
      if (v < bv || (v === bv && Math.abs((owner[j - 1] >= 0 ? owner[j - 1] : 0) - c) < Math.abs((owner[bj - 1] >= 0 ? owner[bj - 1] : 0) - c))) { bv = v; bj = j; }
    }
    if (bj < 0 || bv > 0.45 * H.replace(/ /g, '').length) return;
    let i = m, j = bj;
    while (i > 0) { const b = back[i * (n + 1) + j]; if (b === 0) { i--; j--; } else if (b === 2) i--; else j--; }
    const cs = j, ce = bj - 1;                        // matched stretch of C
    const got = new Map();
    for (let q = cs; q <= ce; q++) if (owner[q] >= 0) got.set(owner[q], (got.get(owner[q]) || 0) + 1);
    const js = [...got].filter(([jj, k]) => k >= 0.5 * Q.W[jj].norm.length).map(([jj]) => jj).sort((x, y) => x - y);
    if (!js.length) return;
    const { labels, wordAt } = labelsOf(js.map((jj) => Q.W[jj].norm));
    const al = ctcAlign(p.lp, p.Tout, labels);
    if (!al) return;
    const t0 = new Array(js.length).fill(-1), t1 = new Array(js.length).fill(-1);
    labels.forEach((_, l) => { const k = wordAt[l]; if (k < 0 || al.first[l] < 0) return; if (t0[k] < 0) t0[k] = al.first[l]; t1[k] = al.last[l]; });
    js.forEach((jj, k) => { if (t0[k] >= 0) words.push({ j: jj, t0: p.a / SR + t0[k] * FRAME + offset, t1: p.a / SR + (t1[k] + 1) * FRAME + offset, rep: 1 }); });
  });
  words.sort((x, y) => x.t0 - y.t0);
  return { kind: 'quran', suras, votes: Object.fromEntries(votes), words, matched: loc.matched, total: hyp.length };
}
function allOpt(Q, a, b) { for (let q = a; q < b; q++) if (!Q.W[q].opt) return false; return true; }
