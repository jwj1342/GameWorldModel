// 按颜色/区域筛选点云，输出包围盒
import { readFileSync } from 'node:fs';
const pts = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const { xyz, rgb } = pts;

function report(name, pred) {
  const sel = [];
  for (let i = 0; i < xyz.length; i++) if (pred(xyz[i], rgb[i])) sel.push(i);
  if (!sel.length) { console.log(name, ': none'); return; }
  const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9], c = [0, 0, 0];
  for (const i of sel) for (let k = 0; k < 3; k++) {
    const v = xyz[i][k];
    mn[k] = Math.min(mn[k], v); mx[k] = Math.max(mx[k], v); c[k] += v;
  }
  console.log(name, 'n=' + sel.length,
    'min', mn.map(v => v.toFixed(2)).join(','),
    'max', mx.map(v => v.toFixed(2)).join(','),
    'ctr', c.map(v => (v / sel.length).toFixed(2)).join(','));
}
const dark = c => c[0] < 70 && c[1] < 70 && c[2] < 70;
report('piano_black(z>2.2)', (p, c) => p[2] > 2.2 && dark(c));
report('black_low(z>2.2,y<1.2)', (p, c) => p[2] > 2.2 && p[1] < 1.2 && dark(c));
report('terracotta_pot', (p, c) => c[0] > 120 && c[0] < 200 && c[1] > 50 && c[1] < 120 && c[2] < 80 && p[2] < 1 && p[2] > -1.5);
report('plant_green', (p, c) => c[1] > c[0] + 15 && c[1] > c[2] + 10 && c[1] > 60 && p[2] < 1.2);
report('rug_dark_green', (p, c) => p[1] < 0.12 && c[1] >= c[0] && c[1] >= c[2] && c[0] < 80 && c[1] < 90);
report('rug_light', (p, c) => p[1] < 0.1 && c[0] > 120 && c[1] > 130 && c[2] > 130);
report('wood_floor', (p, c) => p[1] < 0.08 && c[0] > 150 && c[1] > 100 && c[2] < 120);
report('tv_black(wall x>0.8)', (p, c) => p[0] > 0.8 && dark(c));
report('black_x>0.9,z-1..1', (p, c) => p[0] > 0.9 && p[2] > -1.5 && p[2] < 1 && dark(c));
report('sofa_grey(x<-0.7)', (p, c) => p[0] < -0.7 && c[0] > 80 && c[0] < 140 && Math.abs(c[0] - c[1]) < 15 && Math.abs(c[1] - c[2]) < 15);
report('armchair_bluegrey', (p, c) => c[2] > c[0] && c[2] > 60 && c[0] > 40 && c[0] < 110 && p[2] < -1.2);
report('coffee_table_top', (p, c) => p[1] > 0.3 && p[1] < 0.6 && c[0] > 90 && c[0] < 170 && c[1] > 40 && c[1] < 110 && c[2] < 70 && p[2] > -1.5 && p[2] < 0.5);
report('door_white(z>2.5)', (p, c) => p[2] > 2.3 && c[0] > 180 && c[1] > 175 && c[2] > 165);
report('walls_white', (p, c) => c[0] > 190 && c[1] > 185 && c[2] > 170);
report('curtains', (p, c) => c[0] > 90 && c[0] < 150 && c[1] > 80 && c[1] < 140 && c[2] > 60 && c[2] < 120 && p[2] < -1.5);
report('footstool_blue', (p, c) => c[2] > c[0] + 20 && c[2] > 80 && p[1] < 0.6 && p[2] > -1.5 && p[2] < 0.5);
report('slippers_brown', (p, c) => p[1] < 0.15 && c[0] > 110 && c[1] > 70 && c[1] < 130 && c[2] < 80 && p[2] > -0.5 && p[2] < 1.5);
