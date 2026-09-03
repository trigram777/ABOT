
/* ------------------------------------------------------------------- UI */
function seg(id, opts, get, set) {
  const el = document.getElementById(id);
  el.innerHTML = opts.map(([v, l]) =>
    `<button type="button" data-v="${v}" aria-pressed="${String(get()) === String(v)}">${l}</button>`).join('');
  el.onclick = e => {
    const b = e.target.closest('button'); if (!b) return;
    set(b.dataset.v);
    [...el.children].forEach(c => c.setAttribute('aria-pressed', c === b));
    stage();
  };
}
const clk = m => `${String(((570 + m) / 60) | 0).padStart(2, '0')}:${String((570 + m) % 60).padStart(2, '0')}`;
const opt = (v, l, on) => `<option value="${v}"${on ? ' selected' : ''}>${l}</option>`;

const ACT_WHY = {
  close: 'Close and take the proceeds. Costs a second commission and a fill.',
  short: 'Keep the long and SELL one strike TOWARD the money, holding the spread ' +
         'to expiry. Collects a credit, caps risk at the width, and needs NO exit ' +
         'fill at all — settlement is free.',
  cover: 'Keep the long and SELL one strike FURTHER OUT, holding to expiry. A long ' +
         'spread: smaller credit, but the position can still gain.'
};

function buildUI() {
  seg('sel', H.sel.map((s, i) => [i, s.replace('delta:', 'Δ').replace('price:', '$')]),
    () => S.sel, v => S.sel = +v);
  seg('side', [['auto', 'signal picks'], ['call', 'calls'], ['put', 'puts']],
    () => S.side, v => S.side = v);
  seg('act', [['close', 'close'], ['short', 'SHORT it'], ['cover', 'COVER it']],
    () => S.act, v => { S.act = v; document.getElementById('actwhy').textContent = ACT_WHY[v]; });

  const sl = document.getElementById('slots');
  sl.innerHTML = H.slots.map((m, i) =>
    `<button type="button" data-i="${i}" aria-pressed="true">${clk(m)}</button>`).join('');
  sl.onclick = e => {
    const b = e.target.closest('button'); if (!b) return;
    const i = +b.dataset.i;
    S.slots.has(i) ? S.slots.delete(i) : S.slots.add(i);
    b.setAttribute('aria-pressed', S.slots.has(i)); stage();
  };
  const setSlots = xs => {
    S.slots = new Set(xs);
    [...sl.children].forEach((c, i) => c.setAttribute('aria-pressed', S.slots.has(i)));
    stage();
  };
  document.getElementById('sall').onclick = () => setSlots(H.slots.map((_, i) => i));
  document.getElementById('slate').onclick = () =>
    setSlots(H.slots.map((_, i) => i).filter(i => H.slots[i] >= 375));
  document.getElementById('snone').onclick = () => setSlots([]);

  const mk = (id, none, vals, fmtv, set) => {
    const el = document.getElementById(id);
    el.innerHTML = opt(0, none, true) + vals.map(v => opt(v, fmtv(v))).join('');
    el.onchange = () => { set(+el.value); stage(); };
  };
  mk('l', 'none', [0.35, 0.5, 0.65, 0.8, 0.9], v => v.toFixed(2) + '×', v => S.l = v);
  mk('w', 'none', [1.2, 1.5, 2, 2.5, 3, 4, 5, 7], v => v.toFixed(2) + '×', v => S.w = v);
  mk('tr', 'none', [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], v => v.toFixed(2), v => S.trail = v);
  mk('ck', 'none', [1, 2, 3, 5, 10, 15, 20, 30], v => '+' + v + ' min', v => S.clock = v);

  const xc = document.getElementById('xcol');
  xc.innerHTML = opt('', '— no chart exit —', true) +
    H.columns.map(c => opt(c, c)).join('');
  const xt = document.getElementById('xtf');
  xt.innerHTML = H.tfs.map(t => opt(t, t + 'm', t === 30)).join('');
  const xd = document.getElementById('xdir');
  xd.innerHTML = opt('ge', 'rises to ≥', true) + opt('le', 'falls to ≤');
  const xv = document.getElementById('xval');
  const xrange = () => {
    const m = H.ind.find(z => z.col === xc.value && z.tf === +xt.value);
    document.getElementById('xrange').textContent = xc.value
      ? (m && m.ok ? `${xc.value} @${xt.value}m runs ${m.lo.toFixed(2)} … ${m.hi.toFixed(2)}`
                   : 'not available at this timeframe')
      : 'Exit when the SPX chart itself says so — the first bar after entry that crosses.';
  };
  const xset = () => { S.x = { col: xc.value, tf: +xt.value, dir: xd.value, val: xv.value.trim() };
                       xrange(); stage(); };
  [xc, xt, xd].forEach(e => e.onchange = xset);
  xv.oninput = xset;
  xrange();

  const dsel = document.getElementById('draws');
  dsel.innerHTML = [0, 10, 20, 50].map(v => opt(v, v || 'off', v === S.draws)).join('');
  dsel.onchange = () => { S.draws = +dsel.value; redraw(); };

  const GROUPS = [['band position', ['s_pctb', 'f_pctb', 'pctb_spread']],
                  ['band width', ['s_bandwidth', 'f_bandwidth', 'bandwidth_ratio']],
                  ['gaps', ['gap_low', 'gap_mid', 'gap_high']],
                  ['prior bar / shape', ['prev_range', 'green_red_avg', 'zone', 'slope_pair']]];
  const box = document.getElementById('units');
  box.innerHTML = GROUPS.map(([name, cols], gi) => `
    <div class="unit" data-g="${gi}">
      <div class="top"><h3>${name}</h3>
        <select class="tf" data-g="${gi}">${H.tfs.map(t => opt(t, t + 'm', t === 30)).join('')}</select></div>
      <div class="heads"><span>entry filter</span><span>call ≥</span><span>put ≤</span><span>gate</span><span></span></div>
      ${cols.filter(c => H.columns.includes(c)).map(c => `
        <div class="ind off" data-col="${c}" data-g="${gi}">
          <span class="nm" title="${c}">${c}</span>
          <input type="text" data-k="call" placeholder="—">
          <input type="text" data-k="put" placeholder="—">
          <input type="text" data-k="gate" placeholder="—">
          <select class="cmp" data-k="cmp"><option value="ge">≥</option><option value="le">≤</option></select>
        </div>`).join('')}
      <div class="range" data-g="${gi}"></div></div>`).join('');

  const tfOf = gi => +box.querySelector(`select.tf[data-g="${gi}"]`).value;
  const cfg = (col, gi) => (S.ind[col] ||= { tf: tfOf(gi), call: '', put: '', gate: '', cmp: 'ge' });
  const ranges = gi => {
    const tf = tfOf(gi);
    box.querySelector(`.range[data-g="${gi}"]`).textContent =
      GROUPS[gi][1].filter(c => H.columns.includes(c)).map(c => {
        const m = H.ind.find(z => z.col === c && z.tf === tf);
        return m && m.ok ? `${c} ${m.lo.toFixed(2)}…${m.hi.toFixed(2)}` : `${c} —`;
      }).join('   ');
  };
  GROUPS.forEach((_, gi) => ranges(gi));
  box.addEventListener('input', e => {
    const row = e.target.closest('.ind'); if (!row) return;
    const c = cfg(row.dataset.col, +row.dataset.g);
    c[e.target.dataset.k] = e.target.value.trim(); c.tf = tfOf(+row.dataset.g);
    row.querySelectorAll('input').forEach(inp =>
      inp.classList.toggle('on-' + inp.dataset.k[0], (c[inp.dataset.k] ?? '') !== ''));
    row.classList.toggle('off', !c.call && !c.put && !c.gate);
    stage();
  });
  box.addEventListener('change', e => {
    if (e.target.classList.contains('tf')) {
      const gi = +e.target.dataset.g;
      GROUPS[gi][1].forEach(c => { if (S.ind[c]) S.ind[c].tf = +e.target.value; });
      ranges(gi); stage();
    } else if (e.target.dataset.k === 'cmp') {
      cfg(e.target.closest('.ind').dataset.col, +e.target.closest('.ind').dataset.g)
        .cmp = e.target.value;
      stage();
    }
  });
  document.getElementById('actwhy').textContent = ACT_WHY.close;
}

let ro = null;
addEventListener('resize', () => { clearTimeout(ro); ro = setTimeout(redraw, 140); });
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', redraw);

(async function boot() {
  try {
    D = await decode();
    document.getElementById('boot').remove();
    document.getElementById('app').hidden = false;
    buildUI();
    document.getElementById('go').onclick = apply;
    const ab = document.getElementById('autoup');
    ab.onchange = () => { auto = ab.checked; if (auto) apply(); };
    addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); apply(); } });
    document.getElementById('reset').onclick = () => location.reload();
    apply();
  } catch (err) {
    document.getElementById('boot').textContent = 'could not unpack: ' + err.message;
  }
})();
