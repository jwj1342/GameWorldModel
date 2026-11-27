// 陶盆与绿叶的 z 直方图（严格颜色）
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
function zhist(name, pred) {
  const bins = {};
  let n = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!pred(p, rgb[i])) continue;
    n++;
    const b = Math.round(p[2] * 5) / 5;
    bins[b] = (bins[b] || 0) + 1;
  }
  console.log('---', name, 'n=' + n);
  Object.keys(bins).sort((a, b) => a - b).forEach(k => console.log(k, bins[k]));
}
zhist('pot_strict', (p, c) => c[0] > 125 && c[0] < 185 && c[1] > 65 && c[1] < 115 && c[2] > 35 && c[2] < 85 && c[0] > c[1] + 40 && p[1] < 0.45 && p[1] > -0.05 && p[0] > 0.2 && p[0] < 1.4);
zhist('leaf_strict', (p, c) => c[1] > c[0] + 18 && c[1] > c[2] + 12 && c[1] > 60 && c[1] < 160 && p[1] > 0.2 && p[0] > 0.2 && p[0] < 1.4 && p[2] > -1.8 && p[2] < 2.6);
