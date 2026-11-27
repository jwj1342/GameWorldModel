import fs from 'fs';
const cams = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;
const byName = {};
for (const c of cams) byName[c.name.replace('.JPG', '')] = c;
function rotq(q, v) {
  const [x, y, z, w] = q;
  const dot = x * v[0] + y * v[1] + z * v[2];
  const cx = y * v[2] - z * v[1], cy = z * v[0] - x * v[2], cz = x * v[1] - y * v[0];
  const k = w * w - (x * x + y * y + z * z);
  return [2 * dot * x + k * v[0] + 2 * w * cx, 2 * dot * y + k * v[1] + 2 * w * cy, 2 * dot * z + k * v[2] + 2 * w * cz];
}
function rotqInv(q, v) { return rotq([-q[0], -q[1], -q[2], q[3]], v); }
function project(cam, X) {
  const d = [X[0] - cam.pos[0], X[1] - cam.pos[1], X[2] - cam.pos[2]];
  const v = rotqInv(cam.quat, d);
  const t = Math.tan(cam.fov_deg / 2 * Math.PI / 180);
  const xn = (v[0] / -v[2]) / (t * cam.aspect);
  const yn = (v[1] / -v[2]) / t;
  return [(xn + 1) / 2 * cam.width, (1 - yn) / 2 * cam.height, -v[2]];
}
// usage: node work/check.mjs 'name:x,y,z' 'name2:x,y,z' ... then it prints projections in all views given after --
const args = process.argv.slice(2);
const pts = [];
const views = [];
let mode = 'pts';
for (const a of args) {
  if (a === '--') { mode = 'views'; continue; }
  if (mode === 'pts') {
    const [name, rest] = a.split(':');
    pts.push([name, rest.split(',').map(Number)]);
  } else views.push(a);
}
for (const [name, X] of pts) {
  console.log('== ' + name + ' [' + X.join(', ') + ']');
  for (const vn of views) {
    const c = byName[vn];
    const [px, py, d] = project(c, X);
    console.log('   ' + vn + ': px=' + px.toFixed(0) + ' py=' + py.toFixed(0) + ' d=' + d.toFixed(2));
  }
}
