// 精细物体定位查询
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
function box(name, pred, pct = false) {
  const sel = [];
  for (let i = 0; i < xyz.length; i++) if (pred(xyz[i], rgb[i])) sel.push(xyz[i]);
  if (!sel.length) { console.log(name, ': none'); return; }
  if (pct && sel.length > 20) {
    // 2-98 percentile
    for (let k = 0; k < 3; k++) sel.sort((a, b) => a[k] - b[k]);
    const lo = Math.floor(sel.length * 0.02), hi = Math.ceil(sel.length * 0.98) - 1;
    console.log(name, 'n=' + sel.length,
      'p2', [sel[lo][0], sel[lo][1], sel[lo][2]].map(v => v.toFixed(2)).join(','),
      'p98', [sel[hi][0], sel[hi][1], sel[hi][2]].map(v => v.toFixed(2)).join(','));
    return;
  }
  const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (const p of sel) for (let k = 0; k < 3; k++) { mn[k] = Math.min(mn[k], p[k]); mx[k] = Math.max(mx[k], p[k]); }
  console.log(name, 'n=' + sel.length, 'min', mn.map(v => v.toFixed(2)).join(','), 'max', mx.map(v => v.toFixed(2)).join(','));
}
const dk = c => c[0] < 70 && c[1] < 70 && c[2] < 70;

box('coffee_top', (p, c) => p[1] > 0.32 && p[1] < 0.55 && c[0] > 90 && c[0] < 175 && c[1] > 45 && c[1] < 115 && c[2] < 85 && p[0] > -1.0 && p[0] < 0.1 && p[2] > -0.9 && p[2] < 0.9, true);
box('table_items', (p, c) => p[1] > 0.42 && p[1] < 0.8 && p[0] > -1.0 && p[0] < 0.1 && p[2] > -0.9 && p[2] < 0.9, true);
box('slippers', (p, c) => p[1] < 0.13 && c[0] > 120 && c[1] > 80 && c[2] < 90 && c[0] > c[2] + 40 && p[0] > -0.5 && p[0] < 0.8 && p[2] > -0.3 && p[2] < 1.2, true);
box('rug_light', (p, c) => p[1] < 0.09 && c[0] > 125 && c[1] > 135 && c[2] > 135 && p[0] > -1.2 && p[0] < 1.3 && p[2] > -1.2 && p[2] < 1.8, true);
box('rug_dark', (p, c) => p[1] < 0.09 && c[1] >= c[0] + 5 && c[1] < 95 && c[2] < 95 && c[0] < 75 && p[0] > -1.2 && p[0] < 1.4 && p[2] > -1.2 && p[2] < 2.0, true);
box('subwoofer', (p, c) => dk(c) && p[1] < 0.6 && p[0] > 0.35 && p[0] < 1.05 && p[2] > 0.0 && p[2] < 1.0, true);
box('speaker_south', (p, c) => dk(c) && p[0] > 0.7 && p[2] > -1.45 && p[2] < -0.95, true);
box('speaker_north', (p, c) => dk(c) && p[0] > 0.7 && p[2] > 0.0 && p[2] < 0.5, true);
box('tv_screen', (p, c) => dk(c) && p[0] > 0.95 && p[1] > 0.45 && p[2] > -1.3 && p[2] < 0.3, true);
box('tv_stand', (p, c) => p[0] > 0.9 && p[1] < 0.55 && p[2] > -1.35 && p[2] < 0.35 && !dk(c), true);
box('armchair', (p, c) => c[2] > c[0] + 5 && c[0] > 45 && c[0] < 115 && p[2] < -0.9 && p[2] > -2.2 && p[0] > -0.7 && p[0] < 0.7 && p[1] > 0.1, true);
box('cushion_brownwhite', (p, c) => p[2] < -1.2 && p[2] > -2.1 && p[1] > 0.5 && p[1] < 1.0 && p[0] > -0.5 && p[0] < 0.5 && c[0] > 100 && c[1] > 70, true);
box('footstool_blue', (p, c) => c[2] > c[0] + 15 && c[2] > 75 && p[1] > 0.2 && p[1] < 0.55 && p[0] > -0.9 && p[0] < 0.3 && p[2] > -0.9 && p[2] < 0.6, true);
box('footstool_wood', (p, c) => c[0] > 140 && c[1] > 110 && c[2] < 110 && c[0] > c[2] + 40 && p[1] > 0.05 && p[1] < 0.5 && p[0] > -0.9 && p[0] < 0.3 && p[2] > -0.9 && p[2] < 0.6, true);
box('piano_body', (p, c) => dk(c) && p[2] > 2.35 && p[1] > 0.4, true);
box('piano_bench', (p, c) => dk(c) && p[2] > 2.0 && p[2] < 2.5 && p[1] < 0.6 && p[0] > -1.0 && p[0] < 0.2, true);
box('piano_rug', (p, c) => p[1] < 0.05 && p[2] > 2.2 && p[2] < 3.6 && p[0] > -1.2 && p[0] < 1.2, true);
box('door_white', (p, c) => c[0] > 175 && c[1] > 170 && c[2] > 160 && p[2] > 2.2 && p[2] < 3.4 && p[0] > -1.5 && p[0] < 0.3, true);
box('plant_pot', (p, c) => c[0] > 110 && c[0] < 190 && c[1] > 55 && c[1] < 120 && c[2] < 85 && c[0] > c[1] + 30 && p[1] < 0.45 && p[0] > 0.3 && p[0] < 1.3 && p[2] > -0.2 && p[2] < 1.6, true);
box('window_south', (p, c) => c[0] > 170 && c[1] > 170 && c[2] > 160 && p[2] < -2.1 && p[1] > 0.7, true);
box('window_north', (p, c) => p[2] > 3.5 && p[1] > 0.7, true);
box('alcove_north', (p, c) => c[0] > 150 && c[1] > 145 && c[2] > 135 && p[0] > 0.95 && p[2] > 0.1 && p[2] < 1.2, true);
box('bookshelf_wood', (p, c) => c[0] > 120 && c[1] > 85 && c[2] < 100 && c[0] > c[2] + 30 && p[0] > 0.95 && p[2] > 1.0 && p[2] < 2.3, true);
box('alcove_south', (p, c) => c[0] > 150 && c[1] > 145 && c[2] > 135 && p[0] > 0.95 && p[2] > -2.2 && p[2] < -1.1, true);
box('side_table', (p, c) => c[0] > 130 && c[1] > 100 && c[2] < 110 && p[0] < -0.7 && p[2] > -2.1 && p[2] < -1.3 && p[1] > 0.3, true);
box('sofa_all', (p, c) => p[0] < -0.7 && p[2] > -1.7 && p[2] < 1.0 && p[1] > 0.05 && p[1] < 1.0, true);
box('chaise?', (p, c) => p[0] > -0.7 && p[0] < -0.1 && p[2] > 0.3 && p[2] < 1.3 && p[1] > 0.1 && p[1] < 0.6, true);
box('curtains_south', (p, c) => p[2] < -1.9 && p[1] > 0.3 && c[0] > 80 && c[0] < 160 && c[1] > 70 && c[1] < 150 && c[2] > 50 && c[2] < 130, true);
