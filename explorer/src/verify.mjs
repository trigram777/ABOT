import fs from 'fs'; import zlib from 'zlib';
const R=new URL('../results/', import.meta.url).pathname;
const H=JSON.parse(fs.readFileSync(`${R}/demo_calc_header.json`,'utf8'));
const raw=zlib.gunzipSync(fs.readFileSync(`${R}/demo_calc_blob.bin`));
const ab=raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength);
const n=H.n,wh=H.widths.head,wa=H.widths.aux,HZ=H.horizon;
let o=0;
const head=new Int16Array(ab,o,n*wh); o+=n*wh*2;
const aux=new Float32Array(ab,o,n*wa); o+=n*wa*4;
const Sr={}; for(const k of ['mid','low','short','cover']){Sr[k]=new Int16Array(ab,o,n*HZ);o+=n*HZ*2;}
const ind={}; for(const tf of H.tfs){const nb=H.bars[String(tf)].length,nc=H.columns.length;
  ind[tf]=new Uint8Array(ab,o,H.days.length*nb*nc); o+=H.days.length*nb*nc;}
globalThis.D={n,head,aux,S:Sr,ind,wh,wa};
const src=fs.readFileSync('app.js','utf8');
let body=src.slice(src.indexOf('const day = i =>'), src.indexOf('/* ------------------------------------------- scoring, charts'));
body=body.replace(/^const S = \{[\s\S]*?\n\};\n/m,'');
const mk=new Function('D','H','NA','HZ','S', body+`
  return {run, outcome, entry, live, side, strike, kShort, kCover, settle, px, slot, selOf, day, intrinsic};`);
const S={sel:0,side:'auto',slots:new Set([0,1,2,3,4,5]),l:0,w:0,trail:0,clock:0,
         act:'close',slip:0,x:{col:'',tf:30,dir:'ge',val:''},draws:0,ind:{}};
const E=mk(D,H,-32768,HZ,S);
const show=t=>{const r=E.run([],null);const tot=r.eq.reduce((a,b)=>a+b,0);
  console.log(`${t.padEnd(44)} n=${String(r.n).padStart(6)}  $/entry ${(tot/r.n).toFixed(3).padStart(9)}  ret ${(tot/r.prem*100).toFixed(2).padStart(7)}%  acted ${(100*r.acted/r.n).toFixed(0)}%`);};
show('Δ0.10 all slots, hold to settle, close');
S.sel=2; show('Δ0.35 all slots, hold to settle, close');
S.l=0.65; show('Δ0.35 + L0.65, close');
S.act='short'; show('Δ0.35 + L0.65, SHORT on exit');
S.act='cover'; show('Δ0.35 + L0.65, COVER on exit');
S.act='close'; S.slip=2; show('Δ0.35 + L0.65, close, 2 ticks slip');
S.slip=0; S.l=0; S.slots=new Set([3,4,5]); show('Δ0.35 last 15 min, hold, close');
