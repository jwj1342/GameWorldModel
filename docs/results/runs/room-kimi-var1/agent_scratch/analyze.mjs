// 分析相机朝向和点云布局
import { readFileSync } from 'node:fs';
const cams = JSON.parse(readFileSync('inputs/cameras.json', 'utf8')).views;
const pts = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const xyz = pts.xyz, rgb = pts.rgb;

function quatToMat(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}
// forward = R * (0,0,-1)
for (const c of cams) {
  const m = quatToMat(c.quat);
  const f = [-m[2], -m[5], -m[8]];
  console.log(c.name, 'pos', c.pos.map(v => v.toFixed(2)).join(','),
    'fwd', f.map(v => v.toFixed(2)).join(','));
}
console.log('num points', xyz.length);
// overhead ascii map: x in [-1.6,1.6], z in [-2.6,4.0], cells 0.1m
// mark by max height band of points in cell
const W = 32, H = 66;
const grid = Array.from({ length: H }, () => Array(W).fill(' '));
for (let i = 0; i < xyz.length; i++) {
  const [x, y, z] = xyz[i];
  const cx = Math.floor((x + 1.6) / 0.1), cz = Math.floor((z + 2.6) / 0.1);
  if (cx < 0 || cx >= W || cz < 0 || cz >= H) continue;
  let ch;
  if (y < 0.1) ch = '.';
  else if (y < 0.5) ch = ':';
  else if (y < 1.0) ch = 'o';
  else ch = '#';
  const cur = grid[cz][cx];
  const rank = c => '. :o#'.indexOf(c);
  if (cur === ' ' || rank(ch) > rank(cur)) grid[cz][cx] = ch;
}
console.log('   overhead: x -1.6..1.6 left->right, z -2.6..4.0 top->bottom');
grid.forEach((row, i) => console.log(((-2.6 + i * 0.1).toFixed(1)).padStart(5), row.join('')));
