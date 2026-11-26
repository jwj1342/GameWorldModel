// 北端俯视：x -1.4..0.8, z 2.1..4.0, 0.05m 格，按颜色分类（所有高度）
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const X0 = -1.4, X1 = 0.8, Z0 = 2.1, Z1 = 4.0, S = 0.05;
const W = Math.round((X1 - X0) / S), H = Math.round((Z1 - Z0) / S);
const cells = {};
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i], c = rgb[i];
  const gx = Math.floor((p[0] - X0) / S), gz = Math.floor((p[2] - Z0) / S);
  if (gx < 0 || gx >= W || gz < 0 || gz >= H) continue;
  const s = c[0] + c[1] + c[2];
  let ch = 'o';
  if (s > 540) ch = 'W';
  else if (s < 140) ch = '#';
  else if (c[0] > 140 && c[1] > 100 && c[2] < 120) ch = '.'; // wood
  const k = gz * W + gx;
  cells[k] = cells[k] || {};
  cells[k][ch] = (cells[k][ch] || 0) + 1;
}
for (let gz = 0; gz < H; gz++) {
  let row = '';
  for (let gx = 0; gx < W; gx++) {
    const e = cells[gz * W + gx];
    row += e ? Object.entries(e).sort((a, b) => b[1] - a[1])[0][0] : ' ';
  }
  console.log((Z0 + gz * S).toFixed(2).padStart(6), row);
}
console.log('x from ' + X0 + ' step ' + S);
