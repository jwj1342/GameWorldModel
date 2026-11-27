// 北端高处点（y>1.15，排除钢琴）俯视散布 + 浅色点
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
function scatter(name, pred, X0, X1, Z0, Z1) {
  const S = 0.05, W = Math.round((X1 - X0) / S);
  const cells = {};
  let n = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!pred(p, rgb[i])) continue;
    n++;
    const gx = Math.floor((p[0] - X0) / S), gz = Math.floor((p[2] - Z0) / S);
    if (gx < 0 || gx >= W || gz < 0 || gz > 40) continue;
    cells[gz * 100 + gx] = (cells[gz * 100 + gx] || 0) + 1;
  }
  console.log('---', name, 'n=' + n);
  for (let gz = 0; gz <= 40; gz++) {
    let row = '';
    for (let gx = 0; gx < W; gx++) row += cells[gz * 100 + gx] ? (cells[gz * 100 + gx] > 2 ? '#' : '+') : ' ';
    if (row.trim()) console.log((Z0 + gz * S).toFixed(2), row);
  }
}
scatter('high_y>1.15', (p, c) => p[1] > 1.15 && p[1] < 2.2 && p[2] > 2.2 && p[2] < 4.0 && p[0] > -1.4 && p[0] < 0.9, -1.4, 0.9, 2.2, 4.0);
scatter('door_light_y0.2-1.15', (p, c) => { const s = c[0] + c[1] + c[2]; return s > 400 && p[1] > 0.2 && p[1] < 1.15 && p[2] > 2.2 && p[2] < 3.6 && p[0] > -1.4 && p[0] < 0.9; }, -1.4, 0.9, 2.2, 3.6);
