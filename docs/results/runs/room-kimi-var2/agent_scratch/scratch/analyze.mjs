import fs from 'fs';
const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const xyz = d.xyz, rgb = d.rgb;
const pts = [];
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i];
  if (p[0] > -1.6 && p[0] < 1.6 && p[1] > -0.2 && p[1] < 2.2 && p[2] > -2.6 && p[2] < 4.2) {
    pts.push([p, rgb[i]]);
  }
}
console.log('in-room points:', pts.length);
const x0 = -1.6, z0 = -2.6, cs = 0.2, W = 16, H = 34;
const grid = [];
for (let i = 0; i < H * W; i++) grid.push({ n: 0, maxy: -9 });
for (const [p] of pts) {
  const ix = Math.floor((p[0] - x0) / cs), iz = Math.floor((p[2] - z0) / cs);
  if (ix < 0 || ix >= W || iz < 0 || iz >= H) continue;
  const g = grid[iz * W + ix];
  g.n++;
  g.maxy = Math.max(g.maxy, p[1]);
}
console.log('XZ map: rows = z from ' + z0 + ' step ' + cs + '; cols = x from ' + x0);
for (let iz = 0; iz < H; iz++) {
  let row = '';
  for (let ix = 0; ix < W; ix++) {
    const g = grid[iz * W + ix];
    if (g.n === 0) row += '  .';
    else if (g.n < 4) row += '  ,';
    else {
      const m = g.maxy;
      row += m < 0.15 ? '  _' : m < 0.5 ? '  l' : m < 1.0 ? '  m' : m < 1.6 ? '  h' : '  H';
    }
  }
  console.log((z0 + iz * cs).toFixed(1).padStart(5) + row);
}
let xaxis = '';
for (let ix = 0; ix < W; ix++) xaxis += (x0 + ix * cs).toFixed(1).padStart(3);
console.log('     ' + xaxis);
