// 拟合门板平面：取北端白色点做 PCA
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const sel = [];
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i], c = rgb[i], s = c[0] + c[1] + c[2];
  if (s > 420 && p[2] > 2.3 && p[2] < 3.6 && p[0] > -1.35 && p[0] < 0.5 && p[1] > 0.1 && p[1] < 2.0) sel.push(p);
}
console.log('n=', sel.length);
const m = [0, 0, 0];
for (const p of sel) for (let k = 0; k < 3; k++) m[k] += p[k] / sel.length;
// covariance
const C = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
for (const p of sel) {
  const d = [p[0] - m[0], p[1] - m[1], p[2] - m[2]];
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) C[a][b] += d[a] * d[b] / sel.length;
}
// power iteration for top-2 eigenvectors
function eig(C, deflate) {
  let v = [1, 0.5, 0.3];
  for (let it = 0; it < 200; it++) {
    let w = [0, 0, 0];
    for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) w[a] += C[a][b] * v[b];
    if (deflate) { const d = w[0] * deflate[0] + w[1] * deflate[1] + w[2] * deflate[2]; for (let k = 0; k < 3; k++) w[k] -= d * deflate[k]; }
    const n = Math.hypot(...w); v = w.map(x => x / n);
  }
  return v;
}
const v1 = eig(C);
const v2 = eig(C, v1);
console.log('mean', m.map(v => v.toFixed(2)).join(','));
console.log('v1', v1.map(v => v.toFixed(2)).join(','));
console.log('v2', v2.map(v => v.toFixed(2)).join(','));
// project onto v1/v2 to get extent
let lo1 = 9, hi1 = -9, lo2 = 9, hi2 = -9;
for (const p of sel) {
  const d = [p[0] - m[0], p[1] - m[1], p[2] - m[2]];
  const a = d[0] * v1[0] + d[1] * v1[1] + d[2] * v1[2];
  const b = d[0] * v2[0] + d[1] * v2[1] + d[2] * v2[2];
  lo1 = Math.min(lo1, a); hi1 = Math.max(hi1, a);
  lo2 = Math.min(lo2, b); hi2 = Math.max(hi2, b);
}
console.log('extent v1', lo1.toFixed(2), hi1.toFixed(2), ' v2', lo2.toFixed(2), hi2.toFixed(2));
// also print x/z scatter of these white points, 0.05 cells
const cells = {};
for (const p of sel) {
  const gx = Math.floor((p[0] + 1.35) / 0.05), gz = Math.floor((p[2] - 2.3) / 0.05);
  cells[gz * 100 + gx] = (cells[gz * 100 + gx] || 0) + 1;
}
for (let gz = 0; gz < 26; gz++) {
  let row = '';
  for (let gx = 0; gx < 37; gx++) row += cells[gz * 100 + gx] ? '#' : ' ';
  console.log((2.3 + gz * 0.05).toFixed(2), row);
}
