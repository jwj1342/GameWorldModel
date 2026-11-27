// 地板区域平均颜色采样（0.2m 格）
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
for (let z = -1.9; z < 1.4; z += 0.2) {
  let row = '';
  for (let x = -1.0; x < 1.4; x += 0.2) {
    let n = 0, r = 0, g = 0, b = 0;
    for (let i = 0; i < xyz.length; i++) {
      const p = xyz[i];
      if (p[1] > 0.08 || p[1] < -0.05 || p[0] < x || p[0] >= x + 0.2 || p[2] < z || p[2] >= z + 0.2) continue;
      n++; r += rgb[i][0]; g += rgb[i][1]; b += rgb[i][2];
    }
    row += n < 2 ? '   .   ' : `${String(Math.round(r / n)).padStart(3)},${String(Math.round(g / n)).padStart(3)},${String(Math.round(b / n)).padStart(3)}`;
  }
  console.log(z.toFixed(1).padStart(5), row);
}
