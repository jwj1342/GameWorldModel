// 全面物体定位（修正版：逐轴百分位）
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
function box(name, pred) {
  const sel = [];
  for (let i = 0; i < xyz.length; i++) if (pred(xyz[i], rgb[i])) sel.push(xyz[i]);
  if (sel.length < 3) { console.log(name, ': n=' + sel.length, sel.map(p => p.map(v => v.toFixed(2)).join(',')).join(' | ')); return; }
  const lo = [0, 0, 0], hi = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const vs = sel.map(p => p[k]).sort((a, b) => a - b);
    lo[k] = vs[Math.floor(vs.length * 0.03)];
    hi[k] = vs[Math.min(vs.length - 1, Math.ceil(vs.length * 0.97))];
  }
  console.log(name, 'n=' + sel.length, 'lo', lo.map(v => v.toFixed(2)).join(','), 'hi', hi.map(v => v.toFixed(2)).join(','));
}
const dk = c => c[0] < 70 && c[1] < 70 && c[2] < 70;
const sum = c => c[0] + c[1] + c[2];

box('rug_dark', (p, c) => p[1] < 0.07 && p[1] > -0.05 && c[1] > c[0] && c[1] >= c[2] && sum(c) < 210 && p[0] > -1.2 && p[0] < 1.4 && p[2] > -1.6 && p[2] < 2.0);
box('rug_light', (p, c) => p[1] < 0.07 && p[1] > -0.05 && sum(c) > 360 && sum(c) < 580 && c[2] >= c[0] && p[0] > -1.2 && p[0] < 1.4 && p[2] > -2.0 && p[2] < 1.5);
box('rug_piano', (p, c) => p[1] < 0.07 && p[1] > -0.05 && p[2] > 2.2 && p[2] < 3.7 && p[0] > -1.3 && p[0] < 1.3);
box('slipper_L?', (p, c) => p[1] < 0.12 && p[1] > -0.03 && c[0] > 130 && c[1] > 90 && c[2] > 40 && c[2] < 130 && c[0] > c[2] + 40 && p[0] > -0.6 && p[0] < 0.8 && p[2] > -1.2 && p[2] < 0.8);
box('bowl_metal', (p, c) => p[1] > 0.48 && p[1] < 0.68 && sum(c) > 300 && Math.abs(c[0] - c[1]) < 25 && Math.abs(c[1] - c[2]) < 25 && p[0] > -1.0 && p[0] < -0.3 && p[2] > -0.4 && p[2] < 1.0);
box('carafe_glass', (p, c) => p[1] > 0.48 && p[1] < 0.75 && p[0] > -1.0 && p[0] < -0.3 && p[2] > -0.4 && p[2] < 1.0);
box('door_white2', (p, c) => sum(c) > 480 && p[2] > 2.3 && p[2] < 3.6 && p[0] > -1.4 && p[0] < 0.4 && p[1] > 0.05);
box('door_dark_opening', (p, c) => dk(c) && p[2] > 2.6 && p[0] > -1.4 && p[0] < -0.3 && p[1] > 0.1 && p[1] < 2.0);
box('plantpot_all', (p, c) => c[0] > 100 && c[0] < 190 && c[1] > 50 && c[1] < 125 && c[2] < 90 && c[0] > c[1] + 25 && p[1] < 0.5 && p[1] > -0.05 && p[0] > 0.3 && p[0] < 1.4);
box('plantleaf_all', (p, c) => c[1] > c[0] + 12 && c[1] > c[2] + 8 && c[1] > 55 && p[1] > 0.15 && p[0] > 0.2 && p[0] < 1.4 && p[2] > -1.6 && p[2] < 2.4);
box('alcoveN_white', (p, c) => sum(c) > 420 && p[0] > 1.0 && p[2] > 0.0 && p[2] < 1.3);
box('bookshelfN_wood', (p, c) => c[0] > 110 && c[1] > 75 && c[2] < 110 && c[0] > c[2] + 25 && p[0] > 1.0 && p[2] > 1.1 && p[2] < 2.4);
box('alcoveS_white', (p, c) => sum(c) > 420 && p[0] > 1.0 && p[2] > -2.3 && p[2] < -1.1);
box('tv_dark', (p, c) => dk(c) && p[0] > 0.95 && p[1] > 0.5 && p[1] < 1.45 && p[2] > -1.35 && p[2] < 0.35);
box('spk_stand_S', (p, c) => dk(c) && p[0] > 0.75 && p[2] > -1.4 && p[2] < -0.95 && p[1] < 1.3);
box('spk_stand_N', (p, c) => dk(c) && p[0] > 0.7 && p[2] > 0.05 && p[2] < 0.55 && p[1] < 1.35);
box('subwoofer2', (p, c) => dk(c) && p[1] < 0.55 && p[0] > 0.3 && p[0] < 1.0 && p[2] > -0.1 && p[2] < 0.9);
box('sofa_fabric', (p, c) => c[0] > 70 && c[0] < 150 && Math.abs(c[0] - c[1]) < 18 && Math.abs(c[1] - c[2]) < 18 && p[0] < -0.4 && p[2] > -1.9 && p[2] < 1.2 && p[1] > 0.1 && p[1] < 1.1);
box('chaise_ext', (p, c) => c[0] > 60 && c[0] < 150 && Math.abs(c[0] - c[1]) < 20 && Math.abs(c[1] - c[2]) < 20 && p[0] >= -0.7 && p[0] < 0.0 && p[2] > 0.0 && p[2] < 1.2 && p[1] > 0.05 && p[1] < 0.7);
box('armchair2', (p, c) => c[2] > c[0] && c[2] > 55 && c[0] > 35 && c[0] < 110 && p[2] < -1.2 && p[2] > -2.3 && p[0] > -0.7 && p[0] < 0.6 && p[1] > 0.08);
box('cushion2', (p, c) => p[2] < -1.5 && p[2] > -2.2 && p[1] > 0.5 && p[1] < 1.05 && p[0] > -0.6 && p[0] < 0.4 && (sum(c) > 400 || (c[0] > 90 && c[1] > 60 && c[2] < 80)));
box('footstool3', (p, c) => p[0] > -1.0 && p[0] < 0.2 && p[2] > -1.4 && p[2] < -0.4 && p[1] > 0.1 && p[1] < 0.55);
box('coffee_legs', (p, c) => dk(c) && p[0] > -1.0 && p[0] < -0.35 && p[2] > -0.4 && p[2] < 1.0 && p[1] < 0.5);
box('piano_all', (p, c) => dk(c) && p[2] > 2.3 && p[2] < 3.7 && p[0] > -0.8 && p[0] < 1.0 && p[1] > 0.15);
box('piano_top', (p, c) => dk(c) && p[2] > 2.3 && p[2] < 3.7 && p[0] > -0.8 && p[0] < 1.0 && p[1] > 0.95);
box('bench2', (p, c) => dk(c) && p[2] > 2.1 && p[2] < 2.6 && p[0] > -0.9 && p[0] < 0.3 && p[1] > 0.1 && p[1] < 0.65);
box('sidetable2', (p, c) => p[0] < -0.75 && p[2] > -2.1 && p[2] < -1.3 && p[1] > 0.3 && p[1] < 0.8);
box('window_south2', (p, c) => p[2] < -2.15 && p[1] > 0.6 && p[1] < 2.2 && p[0] > -0.9 && p[0] < 1.2);
box('window_north2', (p, c) => p[2] > 3.55 && p[2] < 4.3 && p[1] > 0.5 && p[1] < 2.4 && p[0] > -0.5 && p[0] < 1.5);
box('curtain_S', (p, c) => p[2] < -1.85 && p[1] > 0.2 && c[0] > 70 && c[0] < 160 && c[1] > 60 && c[1] < 150 && c[2] > 45 && c[2] < 135 && c[0] >= c[2]);
box('curtain_S_wide', (p, c) => p[2] < -1.85 && p[1] > 0.2 && p[1] < 2.2 && sum(c) > 150 && sum(c) < 450);
box('wall_east', (p, c) => p[0] > 1.25 && p[1] > 0.3 && sum(c) > 300);
box('wall_west', (p, c) => p[0] < -1.25 && p[1] > 0.3 && sum(c) > 300);
box('wall_north', (p, c) => p[2] > 3.6 && p[1] > 0.3 && sum(c) > 300 && p[0] < 1.0);
box('wall_south', (p, c) => p[2] < -2.3 && p[1] > 0.3 && sum(c) > 300 && p[0] < 0.5);
box('ceiling', (p, c) => p[1] > 2.2 && sum(c) > 300);
