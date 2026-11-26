import fs from 'fs';
const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const xyz = d.xyz, rgb = d.rgb;

function q(pts, axis, t) {
  const v = pts.map(p => p[axis]).sort((a, b) => a - b);
  return v[Math.max(0, Math.min(v.length - 1, Math.floor(t * (v.length - 1))))];
}
function box(name, x0, x1, y0, y1, z0, z1) {
  const pts = [];
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1 && p[2] >= z0 && p[2] <= z1) pts.push(p);
  }
  if (pts.length < 5) { console.log(name.padEnd(16), 'n=' + pts.length, '(too few)'); return; }
  let sr = 0, sg = 0, sb = 0;
  // mean color
  let cnt = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1 && p[2] >= z0 && p[2] <= z1) {
      sr += rgb[i][0]; sg += rgb[i][1]; sb += rgb[i][2]; cnt++;
    }
  }
  console.log(name.padEnd(16), 'n=' + String(pts.length).padEnd(5),
    'x[' + q(pts, 0, 0.02).toFixed(2) + ',' + q(pts, 0, 0.98).toFixed(2) + ']',
    'y[' + q(pts, 1, 0.02).toFixed(2) + ',' + q(pts, 1, 0.98).toFixed(2) + ']',
    'z[' + q(pts, 2, 0.02).toFixed(2) + ',' + q(pts, 2, 0.98).toFixed(2) + ']',
    'rgb(' + Math.round(sr / cnt) + ',' + Math.round(sg / cnt) + ',' + Math.round(sb / cnt) + ')');
}

box('armchair', -0.8, 0.6, 0.05, 1.3, -2.45, -1.35);
box('armchair-low', -0.8, 0.6, 0.05, 0.6, -2.45, -1.35);
box('sofa', -1.75, -0.65, 0.05, 1.2, -2.0, 1.7);
box('sofa-zsplit-a', -1.75, -0.65, 0.05, 1.2, -2.0, -0.5);
box('sofa-zsplit-b', -1.75, -0.65, 0.05, 1.2, -0.5, 0.5);
box('sofa-zsplit-c', -1.75, -0.65, 0.05, 1.2, 0.5, 1.7);
box('coffee_table', -1.05, -0.25, 0.05, 0.6, -0.3, 0.7);
box('footstool', -0.85, 0.05, 0.03, 0.5, -1.35, -0.45);
box('subwoofer', 0.0, 0.75, 0.02, 0.7, -1.25, -0.35);
box('tv', 0.75, 1.45, 0.6, 1.6, -1.35, 0.1);
box('tv_stand', 0.75, 1.45, 0.0, 0.6, -1.35, 0.2);
box('speaker_L', 0.55, 1.15, 0.5, 1.4, -1.75, -1.0);
box('speaker_R', 0.8, 1.4, 0.5, 1.4, 0.55, 1.25);
box('plant', 0.35, 1.05, 0.0, 1.6, 0.55, 1.25);
box('piano', 0.15, 1.5, 0.05, 1.4, 0.95, 2.45);
box('piano-hi', 0.15, 1.5, 0.6, 1.4, 0.95, 2.45);
box('piano_bench', -0.3, 0.5, 0.05, 0.7, 0.9, 1.9);
box('door', -1.5, -0.4, 0.0, 2.2, 2.3, 3.5);
box('far-end', -1.5, 1.5, 0.0, 2.4, 3.0, 4.2);
box('slippers', -0.35, 0.45, 0.0, 0.2, -1.15, -0.35);
box('alcove-games', 0.85, 1.6, 0.3, 2.3, -2.55, -1.75);
box('alcove-books', 0.85, 1.6, 0.3, 2.3, 0.3, 1.3);
box('bookcase-mid', 0.85, 1.6, 0.3, 2.3, -1.8, -1.0);
box('negz-wall', -1.5, 1.5, 0.3, 2.4, -2.7, -2.25);
box('wall-xneg', -1.7, -1.25, 0.3, 2.4, -2.3, 3.9);
box('wall-xpos-a', 1.2, 1.65, 1.6, 2.4, -2.3, 1.0);
box('wall-xpos-b', 1.3, 1.65, 0.3, 2.4, 1.0, 2.4);
box('ceiling-ish', -1.5, 1.5, 2.2, 3.0, -2.4, 3.9);
