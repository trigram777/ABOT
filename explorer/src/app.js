const H = JSON.parse(document.getElementById('hdr').textContent);
const NA = -32768, HZ = H.horizon;
let D = null, trials = 0;

/** The blob arrives one of two ways and ONE decoder serves both, so the two
 *  builds cannot drift in how they read the format.
 *
 *  SPLIT  (`build.py --split`) -- the page carries a URL and fetches the gzip
 *         stream, which begins inflating while it is still downloading and is
 *         cached separately from the HTML. Base64 is a flat 33% tax and this
 *         does not pay it. Needs a real origin: `fetch` against a file:// URL
 *         is blocked in Chrome.
 *  INLINE (default) -- the bytes sit in the document as base64. Larger, and it
 *         must be parsed before anything runs, but it opens from file:// and
 *         from anywhere a second request is refused.
 *
 *  The split file is named `.bin`, NOT `.gz`, deliberately. A server seeing a
 *  `.gz` extension may send it with `Content-Encoding: gzip`; the browser then
 *  inflates it in transit and DecompressionStream is handed something that is
 *  no longer gzip. Keeping the extension neutral keeps the encoding ours. */
async function decode() {
  const url = document.getElementById('hdr').dataset.blob;
  let stream;
  if (url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url}: ${res.status} ${res.statusText}`);
    stream = res.body;
  } else {
    const b64 = document.getElementById('blob').textContent.trim();
    stream = new Blob([Uint8Array.from(atob(b64), c => c.charCodeAt(0))]).stream();
  }
  const buf = await new Response(
    stream.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  const n = H.n, wh = H.widths.head, wa = H.widths.aux;
  let o = 0;
  const head = new Int16Array(buf, o, n * wh); o += n * wh * 2;
  const aux = new Float32Array(buf, o, n * wa); o += n * wa * 4;
  const S = {};
  for (const k of ['mid', 'low', 'short', 'cover']) {
    S[k] = new Int16Array(buf, o, n * HZ); o += n * HZ * 2;
  }
  const ind = {};
  for (const tf of H.tfs) {
    const nb = H.bars[String(tf)].length, nc = H.columns.length;
    ind[tf] = new Uint8Array(buf, o, H.days.length * nb * nc);
    o += H.days.length * nb * nc;
  }
  return { n, head, aux, S, ind, wh, wa };
}

const day = i => D.head[i * D.wh], slot = i => D.head[i * D.wh + 1];
const side = i => D.head[i * D.wh + 2], selOf = i => D.head[i * D.wh + 3];
const entry = i => D.head[i * D.wh + 4] / 100;
const live = i => D.head[i * D.wh + 6];
const strike = i => D.aux[i * D.wa], kShort = i => D.aux[i * D.wa + 1];
const kCover = i => D.aux[i * D.wa + 2], settle = i => D.aux[i * D.wa + 3];
const px = (k, i, t) => { const v = D.S[k][i * HZ + t]; return v === NA ? NaN : v / 100; };
/** Cash settlement of one option, per share. Free and automatic — the whole
 *  reason a conversion is interesting is that it needs no fill. */
const intrinsic = (i, K) => side(i) === 0 ? Math.max(0, settle(i) - K)
  : Math.max(0, K - settle(i));
const tick = p => p < 3 ? 0.05 : 0.10;
const snapDown = p => Math.floor(p / tick(p) + 1e-9) * tick(p);
const fee = p => p >= 1 ? 1.63 : 1.54;

const S = {
  sel: 0, side: 'auto', slots: new Set([0, 1, 2, 3, 4, 5]),
  l: 0, w: 0, trail: 0, clock: 0, act: 'close', slip: 0,
  x: { col: '', tf: 30, dir: 'ge', val: '' }, draws: 20, ind: {}
};

/* ------------------------------------------------------------ chart exits */
function indAt(tf, d, bi, ci) {
  const nb = H.bars[String(tf)].length, nc = H.columns.length;
  const v = D.ind[tf][(d * nb + bi) * nc + ci];
  return v === 255 ? NaN : v;
}
function toByte(m, v) {
  return Math.max(0, Math.min(254, Math.round((v - m.lo) / (m.hi - m.lo) * 254)));
}
function metaOf(col, tf) { return H.ind.find(m => m.col === col && m.tf === tf); }

/** First minute AFTER entry at which the chart condition holds, or Infinity.
 *  Bars are absolute minutes from 09:30, so a bar at or before the entry slot
 *  is information the entry already had and cannot be an exit. */
function chartExit(i) {
  const c = S.x;
  if (!c.col || c.val === '' || isNaN(+c.val)) return Infinity;
  const m = metaOf(c.col, c.tf);
  if (!m || !m.ok) return Infinity;
  const ci = H.columns.indexOf(c.col), bars = H.bars[String(c.tf)];
  const lvl = toByte(m, +c.val), s = H.slots[slot(i)], d = day(i);
  for (let b = 0; b < bars.length; b++) {
    if (bars[b] <= s) continue;
    const v = indAt(c.tf, d, b, ci);
    if (isNaN(v)) continue;
    if (c.dir === 'ge' ? v >= lvl : v <= lvl) return bars[b] - s;
  }
  return Infinity;
}

/* ------------------------------------------------------------- the exit */
function outcome(i) {
  const e = entry(i), n = Math.min(live(i), HZ);
  const lStop = S.l ? snapDown(S.l * e) : 0;
  const wLim = S.w ? snapDown(S.w * e) : 0;
  let hw = e, t = -1, why = '';
  const tX = chartExit(i), tC = S.clock || Infinity;
  for (let k = 1; k < n; k++) {
    const m = px('mid', i, k), lo = px('low', i, k);
    // precedence inside one minute: adverse first (exits.py's convention)
    if (lStop && !isNaN(lo) && lo <= lStop) { t = k; why = 'L'; break; }
    if (S.trail && !isNaN(lo) && hw > e) {
      const lvl = snapDown(S.trail * hw);
      if (lo <= lvl) { t = k; why = 'T'; break; }
    }
    if (k >= tX) { t = k; why = 'X'; break; }
    if (k >= tC) { t = k; why = 'C'; break; }
    if (wLim && !isNaN(m) && m >= wLim) { t = k; why = 'W'; break; }
    if (!isNaN(m)) hw = Math.max(hw, m);
  }
  const K = strike(i);
  if (t < 0) {                                   // never fired — settles free
    return { pnl: (intrinsic(i, K) - e) * 100 - fee(e), why: 'settle', acted: false };
  }
  // what the exit is worth, per share
  let proceeds;
  if (why === 'L') proceeds = Math.max(0, lStop - S.slip * tick(e));
  else if (why === 'T') proceeds = Math.max(0, snapDown(S.trail * hw) - S.slip * tick(e));
  else if (why === 'W') proceeds = wLim;
  else proceeds = px('mid', i, t);
  if (isNaN(proceeds)) proceeds = px('mid', i, t);
  if (isNaN(proceeds)) return { pnl: (intrinsic(i, K) - e) * 100 - fee(e),
                                why: 'dark', acted: false };

  if (S.act === 'close') {
    return { pnl: (proceeds - e) * 100 - fee(e) - fee(proceeds), why, acted: true };
  }
  // SHORT or COVER: keep the long, SELL a neighbour, hold the spread to expiry.
  // No exit fill on the long at all — settlement is free — which is the point.
  //
  // **THE CREDIT IS TAKEN AT t+1, NOT t, AND THAT IS THE WHOLE RESULT.** A
  // minute's quote is the book at the START of that minute, while the stop
  // triggers on the intra-minute LOW — so the neighbour's price at t existed
  // BEFORE the trigger. For a stopped call, spot fell during t, so selling at
  // t's open collects a credit the trigger had already destroyed. Measured:
  // pricing at t rather than t+1 was worth **$62.73 an entry against a $5.80
  // result** — more than the entire edge. It only bites the conversions,
  // because they are the one exit whose FILL is
  // on a different contract from the TRIGGER.
  const which = S.act === 'short' ? 'short' : 'cover';
  const Kc = S.act === 'short' ? kShort(i) : kCover(i);
  const tc = t + 1;
  const credit = tc < Math.min(live(i), HZ) ? px(which, i, tc) : NaN;
  if (isNaN(credit) || isNaN(Kc)) {
    return { pnl: (proceeds - e) * 100 - fee(e) - fee(proceeds), why: why + '!', acted: true };
  }
  const val = intrinsic(i, K) - intrinsic(i, Kc);   // long minus short at expiry
  return { pnl: (credit - e + val) * 100 - fee(e) - fee(credit), why, acted: true };
}

/* --------------------------------------------------------------- the rule */
function compile() {
  const out = [];
  for (const [key, cfg] of Object.entries(S.ind)) {
    const m = metaOf(key, cfg.tf);
    if (!m || !m.ok) continue;
    const num = v => (v === '' || v == null || isNaN(+v)) ? null : toByte(m, +v);
    const c = num(cfg.call), p = num(cfg.put), g = num(cfg.gate);
    if (c === null && p === null && g === null) continue;
    out.push({ ci: H.columns.indexOf(key), tf: cfg.tf, c, p, g,
               cmp: cfg.cmp === 'le' ? -1 : 1 });
  }
  return out;
}
/** The chart value AT the entry slot — the last bar at or before it. */
function atEntry(i, tf, ci) {
  const bars = H.bars[String(tf)], s = H.slots[slot(i)], d = day(i);
  let v = NaN;
  for (let b = 0; b < bars.length && bars[b] <= s; b++) {
    const x = indAt(tf, d, b, ci);
    if (!isNaN(x)) v = x;
  }
  return v;
}
function decide(i, rule, srcDay) {
  if (!rule.length) return S.side === 'call' ? 1 : S.side === 'put' ? 2 : 3;
  let votes = 0, any = false;
  for (const r of rule) {
    const bars = H.bars[String(r.tf)], s = H.slots[slot(i)];
    let v = NaN;
    for (let b = 0; b < bars.length && bars[b] <= s; b++) {
      const x = indAt(r.tf, srcDay, b, r.ci);
      if (!isNaN(x)) v = x;
    }
    if (isNaN(v)) continue;
    if (r.g !== null && (r.cmp > 0 ? v >= r.g : v <= r.g)) return 0;
    if (r.c !== null && v >= r.c) { votes++; any = true; }
    if (r.p !== null && v <= r.p) { votes--; any = true; }
  }
  if (S.side === 'call') return (votes > 0 || !any) ? 1 : 0;
  if (S.side === 'put') return (votes < 0 || !any) ? 2 : 0;
  if (votes > 0) return 1;
  if (votes < 0) return 2;
  return 0;
}

function run(rule, perm) {
  const nD = H.days.length, eq = new Float64Array(nD);
  const xs = []; let n = 0, acted = 0, prem = 0;
  for (let i = 0; i < D.n; i++) {
    if (selOf(i) !== S.sel || !S.slots.has(slot(i))) continue;
    const d = day(i);
    const want = decide(i, rule, perm ? perm[d] : d);
    if (want === 0) continue;
    const me = side(i) === 0 ? 1 : 2;
    if (want !== 3 && want !== me) continue;
    const o = outcome(i);
    eq[d] += o.pnl; n++; prem += entry(i) * 100;
    if (o.acted) acted++;
    if (!perm) xs.push(o.pnl / (entry(i) * 100) + 1);
  }
  return { eq, n, xs, acted, prem };
}

/* ------------------------------------------- scoring, charts, and the null */
/* --------------------------------------------------------------- scoring */
function score(eq, n) {
  const nD = eq.length;
  let tot = 0, sq = 0, pos = 0, live = 0;
  for (let i = 0; i < nD; i++) { tot += eq[i]; }
  const mean = tot / nD;
  for (let i = 0; i < nD; i++) { const d = eq[i] - mean; sq += d * d; if (eq[i] > 0) pos++; if (eq[i] !== 0) live++; }
  const sd = Math.sqrt(sq / (nD - 1));
  let peak = 0, cum = 0, dd = 0;
  const curve = new Float64Array(nD);
  for (let i = 0; i < nD; i++) { cum += eq[i]; curve[i] = cum; peak = Math.max(peak, cum); dd = Math.min(dd, cum - peak); }
  const srt = Array.from(eq).sort((a, b) => b - a);
  let half = -1, acc = 0;
  if (tot > 0) for (let i = 0; i < nD; i++) { acc += srt[i]; if (acc >= tot / 2) { half = i + 1; break; } }
  return { curve, total: tot, perDay: mean, perTrade: n ? tot / n : 0,
    sharpe: sd ? mean / sd * Math.sqrt(252) : 0, dd, pos: live ? pos / live : 0,
    days: live, half, n };
}

/* ---------------------------------------------------------------- render */
function fmt(v, d = 0) { return v.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d }); }
function css(v) { return getComputedStyle(document.body).getPropertyValue(v).trim(); }

function tiles(s, base) {
  const sign = v => v > 0 ? 'pos' : v < 0 ? 'neg' : '';
  const T = [
    ['per day', '$' + fmt(s.perDay, 1), s.n.toLocaleString() + ' entries', sign(s.perDay)],
    ['return on premium', (s.ret * 100).toFixed(1) + '%', 'per entry', sign(s.ret)],
    ['break-even', s.be >= 16 ? '16+ ticks' : s.be.toFixed(1) + ' ticks',
      'slippage per stop', s.be >= 2 ? 'pos' : s.be > 0 ? '' : 'neg'],
    ['sharpe', fmt(s.sharpe, 2), 'daily, ann.', sign(s.sharpe)],
    ['positive days', (s.pos * 100).toFixed(1) + '%', '', ''],
    ['max drawdown', '$' + fmt(s.dd), '', s.dd < 0 ? 'neg' : ''],
    ['days to half', s.half < 0 ? '—' : s.half, s.half > 0 && s.half < 20 ? 'tail-dependent' : '', s.half > 0 && s.half < 20 ? 'neg' : ''],
    ['vs permuted', (base === null ? '—' : (s.total > base ? '+' : '') + '$' + fmt(s.total - base)), base === null ? '' : 'above the null mean', base === null ? '' : sign(s.total - base)]
  ];
  document.getElementById('tiles').innerHTML = T.map(([k, v, n, c]) =>
    `<div class="tile"><span class="k">${k}</span><span class="v ${c}">${v}</span><span class="n">${n}</span></div>`).join('');
}

/** Size a canvas for the device pixel ratio, ONCE per render, idempotently.
 *
 *  The bug this replaces: reading `cv.height` to get the logical height and
 *  then writing `cv.height = that * dpr`. After the first render `cv.height`
 *  IS the scaled value, so every subsequent render multiplied by `dpr` again
 *  — the chart grew by a factor of dpr per update until it broke. The logical
 *  height has to come from somewhere that does not change, so it is captured
 *  from the markup on first use and read from `dataset` forever after. */
function fit(cv) {
  const dpr = devicePixelRatio || 1;
  if (!cv.dataset.h) cv.dataset.h = cv.getAttribute('height') || cv.height;
  const H = +cv.dataset.h;
  const W = Math.max(1, cv.clientWidth || cv.parentElement.clientWidth);
  cv.style.height = H + 'px';
  cv.width = Math.round(W * dpr);
  cv.height = Math.round(H * dpr);
  const g = cv.getContext('2d');
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  return { g, W, H };
}

function line(cv, curve, band) {
  const { g, W, H: Ht } = fit(cv);
  const pad = { l: 62, r: 10, t: 10, b: 22 }, w = W - pad.l - pad.r, h = Ht - pad.t - pad.b;
  const n = curve.length;
  let lo = 0, hi = 0;
  for (const v of curve) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  if (band) for (let i = 0; i < n; i++) { lo = Math.min(lo, band.lo[i]); hi = Math.max(hi, band.hi[i]); }
  if (hi === lo) hi = lo + 1;
  const pd = 0.06 * (hi - lo); lo -= pd; hi += pd;
  const X = i => pad.l + i / (n - 1) * w, Y = v => pad.t + (1 - (v - lo) / (hi - lo)) * h;
  g.strokeStyle = css('--line-soft'); g.fillStyle = css('--text-faint');
  g.font = '10px "IBM Plex Mono",monospace'; g.textAlign = 'right';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText('$' + fmt(v), pad.l - 7, y + 3);
  }
  g.setLineDash([3, 3]); g.strokeStyle = css('--text-faint'); g.beginPath();
  g.moveTo(pad.l, Y(0)); g.lineTo(W - pad.r, Y(0)); g.stroke(); g.setLineDash([]);
  if (band) {
    g.fillStyle = css('--null') + '44'; g.beginPath(); g.moveTo(X(0), Y(band.lo[0]));
    for (let i = 1; i < n; i++) g.lineTo(X(i), Y(band.lo[i]));
    for (let i = n - 1; i >= 0; i--) g.lineTo(X(i), Y(band.hi[i]));
    g.closePath(); g.fill();
  }
  g.strokeStyle = css('--brass'); g.lineWidth = 1.6; g.beginPath();
  for (let i = 0; i < n; i++) i ? g.lineTo(X(i), Y(curve[i])) : g.moveTo(X(i), Y(curve[i]));
  g.stroke();
  g.textAlign = 'left'; g.fillStyle = css('--text-faint');
  g.fillText(H.days[0].slice(0, 7), pad.l, Ht - 6);
  g.textAlign = 'right'; g.fillText(H.days[n - 1].slice(0, 7), W - pad.r, Ht - 6);
}

function hist(cv, xs) {
  const { g, W, H: Ht } = fit(cv);
  const edges = [0, .1, .25, .5, .75, .9, 1, 1.25, 1.5, 2, 3, 5, 1e9];
  const labs = ['0', '.1', '.25', '.5', '.75', '.9', '1', '1.25', '1.5', '2', '3', '5+'];
  const c = new Array(edges.length - 1).fill(0);
  for (const x of xs) { let k = 0; while (k < edges.length - 2 && x >= edges[k + 1]) k++; c[k]++; }
  const mx = Math.max(1, ...c), pad = { l: 8, r: 8, t: 8, b: 20 };
  const w = (W - pad.l - pad.r) / c.length, h = Ht - pad.t - pad.b;
  g.font = '10px "IBM Plex Mono",monospace'; g.textAlign = 'center';
  for (let i = 0; i < c.length; i++) {
    const bh = c[i] / mx * h, x = pad.l + i * w;
    g.fillStyle = edges[i] < 1 ? css('--rust') : css('--jade');
    g.globalAlpha = .82; g.fillRect(x + 1.5, pad.t + h - bh, w - 3, bh); g.globalAlpha = 1;
    g.fillStyle = css('--text-faint');
    g.fillText(labs[i] + '×', x + w / 2, Ht - 6);
    if (c[i] && bh > 13) { g.fillStyle = css('--panel');
      g.fillText(((c[i] / xs.length) * 100).toFixed(0) + '%', x + w / 2, pad.t + h - bh + 11); }
  }
}

/* ------------------------------------------------------------- the null */


/** Permute WHICH DAY's chart each day reads, bar for bar. Holds the trade
 *  count, the call/put mix and the fat tail; removes only the information. */
function permutation(seed) {
  const nD = H.days.length;
  let s = seed >>> 0 || 1;
  const rnd = () => (s ^= s << 13, s ^= s >>> 17, s ^= s << 5, (s >>> 0) / 4294967296);
  const p = new Int32Array(nD);
  for (let i = 0; i < nD; i++) p[i] = i;
  for (let i = nD - 1; i > 0; i--) { const j = (rnd() * (i + 1)) | 0;[p[i], p[j]] = [p[j], p[i]]; }
  return p;
}
function nullBand(rule, draws) {
  if (!draws || !rule.length) return null;
  const nD = H.days.length, all = [];
  for (let d = 0; d < draws; d++) {
    const r = run(rule, permutation(d * 2654435761 + 12345));
    let cum = 0; const c = new Float64Array(nD);
    for (let i = 0; i < nD; i++) { cum += r.eq[i]; c[i] = cum; }
    all.push(c);
  }
  const lo = new Float64Array(nD), hi = new Float64Array(nD), col = new Float64Array(draws);
  let mean = 0;
  for (let i = 0; i < nD; i++) {
    for (let d = 0; d < draws; d++) col[d] = all[d][i];
    const s = Array.from(col).sort((a, b) => a - b);
    lo[i] = s[Math.floor(.05 * (draws - 1))]; hi[i] = s[Math.ceil(.95 * (draws - 1))];
  }
  for (let d = 0; d < draws; d++) mean += all[d][nD - 1];
  return { lo, hi, mean: mean / draws };
}

/* ------------------------------------------------- staging and applying */
let pending = null, staged = 0, auto = false;
function stage() {
  staged++;
  const b = document.getElementById('go');
  b.classList.add('pending'); b.textContent = `update (${staged})`;
  if (auto) apply();
}
/** How many ticks of slippage on every stop before this returns nothing.
 *  The figure that makes two policies comparable independently of how often
 *  they act. Solved by bisection on the real run. */
function breakEven(rule) {
  const keep = S.slip;
  const at = k => { S.slip = k; const r = run(rule, null);
                    return r.eq.reduce((a, b) => a + b, 0); };
  if (at(0) <= 0) { S.slip = keep; return 0; }
  let lo = 0, hi = 1;
  while (at(hi) > 0 && hi < 16) hi *= 2;
  if (at(hi) > 0) { S.slip = keep; return hi; }
  for (let k = 0; k < 14; k++) { const m = (lo + hi) / 2; (at(m) > 0 ? lo = m : hi = m); }
  S.slip = keep; return (lo + hi) / 2;
}
function apply() {
  clearTimeout(pending);
  pending = setTimeout(() => {
    const rule = compile();
    const r = run(rule, null);
    const s = score(r.eq, r.n);
    s.ret = r.prem ? (s.total / r.prem) : 0;
    s.acted = r.n ? r.acted / r.n : 0;
    s.be = breakEven(rule);
    const band = nullBand(rule, S.draws);
    tiles(s, band ? band.mean : null);
    line(document.getElementById('eq'), s.curve, band);
    hist(document.getElementById('hist'), r.xs);
    document.getElementById('why').textContent =
      `${r.n.toLocaleString()} entries, ${(s.acted * 100).toFixed(0)}% of them ` +
      `exited on a rule rather than settling. Bars left of 1.0× lost money.`;
    if (staged) { trials++; document.getElementById('ntrials').textContent = trials; }
    staged = 0;
    const b = document.getElementById('go');
    b.classList.remove('pending'); b.textContent = 'update';
  }, 10);
}
function redraw() { const k = staged; staged = 0; apply(); staged = k; }
