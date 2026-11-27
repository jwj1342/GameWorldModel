import fs from 'fs';
const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const xyz = d.xyz, rgb = d.rgb;

function box(name, pred) {
  let mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9], n = 0;
  const pts = [];
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i], c = rgb[i];
    if (!pred(p, c)) continue;
    n++;
    pts.push(p);
    for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  }
  if (!n) { console.log(name, ': none'); return pts; }
  // robust bounds: 5-95 percentile per axis
  const q = (axis, t) => { const v = pts.map(p => p[axis]).sort((a, b) => a - b); return v[Math.floor(t * (v.length - 1))]; };
  console.log(name, 'n=' + n,
    'x[' + q(0, 0.03).toFixed(2) + ',' + q(0, 0.97).toFixed(2) + ']',
    'y[' + q(1, 0.03).toFixed(2) + ',' + q(1, 0.97).toFixed(2) + ']',
    'z[' + q(2, 0.03).toFixed(2) + ',' + q(2, 0.97).toFixed(2) + ']');
  return pts;
}

const dark = (c) => c[0] + c[1] + c[2] < 140;
console.log('--- very dark points (TV / piano / speakers), by region ---');
box('dark, y 0.5-1.5, x>0.7', (p, c) => dark(c) && p[1] > 0.5 && p[1] < 1.5 && p[0] > 0.7);
box('dark, y 0.5-1.5, x<0.7, z<0', (p, c) => dark(c) && p[1] > 0.5 && p[1] < 1.5 && p[0] <= 0.7 && p[2] < 0);
box('dark, y 0.5-1.5, x<0.7, z>0', (p, c) => dark(c) && p[1] > 0.5 && p[1] < 1.5 && p[0] <= 0.7 && p[2] > 0);
box('dark, y 0.2-0.6 (piano body?)', (p, c) => dark(c) && p[1] > 0.2 && p[1] < 0.6);

console.log('--- XZ histogram of dark points y 0.4-1.6 ---');
const cells = {};
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i], c = rgb[i];
  if (!dark(c) || p[1] < 0.4 || p[1] > 1.6) continue;
  const k = Math.round(p[0] * 2) / 2 + ',' + Math.round(p[2] * 2) / 2;
  cells[k] = (cells[k] || 0) + 1;
}
const entries = Object.entries(cells).sort((a, b) => b[1] - a[1]).slice(0, 25);
for (const [k, v] of entries) console.log('  (x,z)=(' + k + ') n=' + v);

console.log('--- floor band y -0.02..0.06, color classes ---');
function floorClass(name, pred) {
  const cells = {};
  let n = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i], c = rgb[i];
    if (p[1] < -0.02 || p[1] > 0.06) continue;
    if (!pred(c)) continue;
    n++;
    const k = Math.round(p[0] * 2.5) / 2.5 + ',' + Math.round(p[2] * 2.5) / 2.5;
    cells[k] = (cells[k] || 0) + 1;
  }
  console.log(name, 'n=' + n);
  const e = Object.entries(cells).sort((a, b) => b[1] - a[1]).slice(0, 14);
  console.log('   ', e.map(([k, v]) => k + ':' + v).join('  '));
}
floorClass('dark-green rug', (c) => c[1] > c[0] && c[1] > c[2] && c[1] < 110 && c[1] > 30);
floorClass('light grey rug', (c) => c[0] > 120 && c[1] > 120 && c[2] > 120 && Math.abs(c[0] - c[2]) < 40);
floorClass('warm wood floor', (c) => c[0] > 100 && c[0] > c[2] + 40 && c[1] > c[2]);
