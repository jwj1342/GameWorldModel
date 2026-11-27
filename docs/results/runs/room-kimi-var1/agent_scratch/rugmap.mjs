// 地面材质 ASCII 图：x -1.3..1.5, z -2.2..2.0, 0.05m 格
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const X0 = -1.3, X1 = 1.5, Z0 = -2.2, Z1 = 2.0, S = 0.05;
const W = Math.round((X1 - X0) / S), H = Math.round((Z1 - Z0) / S);
const cnt = {};
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i], c = rgb[i];
  if (p[1] > 0.09 || p[1] < -0.06) continue;
  const gx = Math.floor((p[0] - X0) / S), gz = Math.floor((p[2] - Z0) / S);
  if (gx < 0 || gx >= W || gz < 0 || gz >= H) continue;
  const s = c[0] + c[1] + c[2];
  let ch = '?';
  if (c[1] > c[0] && c[1] >= c[2] && s < 220) ch = 'D';       // dark green
  else if (s > 380 && c[2] >= c[0] - 10) ch = 'L';            // light grey/blue
  else if (c[0] > 140 && c[1] > 95 && c[2] < 130 && c[0] > c[2] + 40) ch = '.'; // wood
  else if (s < 150) ch = 'B';                                  // black-ish
  const k = gz * W + gx;
  cnt[k] = cnt[k] || {};
  cnt[k][ch] = (cnt[k][ch] || 0) + 1;
}
for (let gz = 0; gz < H; gz++) {
  let row = '';
  for (let gx = 0; gx < W; gx++) {
    const e = cnt[gz * W + gx];
    if (!e) { row += ' '; continue; }
    row += Object.entries(e).sort((a, b) => b[1] - a[1])[0][0];
  }
  console.log((Z0 + gz * S).toFixed(2).padStart(6), row);
}
console.log('x axis: ' + X0 + ' .. ' + X1 + ' (0.05/cell)');
