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
function rotqInv(q, v) { return rotq([ -q[0], -q[1], -q[2], q[3] ], v); }

function project(cam, X) {
  const d = [X[0] - cam.pos[0], X[1] - cam.pos[1], X[2] - cam.pos[2]];
  const v = rotqInv(cam.quat, d);
  const t = Math.tan(cam.fov_deg / 2 * Math.PI / 180);
  const xn = (v[0] / -v[2]) / (t * cam.aspect);
  const yn = (v[1] / -v[2]) / t;
  return [(xn + 1) / 2 * cam.width, (1 - yn) / 2 * cam.height, -v[2]];
}
function ray(cam, px, py) {
  const t = Math.tan(cam.fov_deg / 2 * Math.PI / 180);
  const xn = px / cam.width * 2 - 1, yn = 1 - py / cam.height * 2;
  const d = [xn * t * cam.aspect, yn * t, -1];
  const dir = rotq(cam.quat, d);
  const L = Math.hypot(...dir);
  return { o: cam.pos, d: dir.map(v => v / L) };
}
function triangulate(r1, r2) {
  // closest points between two rays
  const [o1, d1] = [r1.o, r1.d], [o2, d2] = [r2.o, r2.d];
  const w = [o1[0] - o2[0], o1[1] - o2[1], o1[2] - o2[2]];
  const a = d1[0]**2 + d1[1]**2 + d1[2]**2, b = d1[0]*d2[0]+d1[1]*d2[1]+d1[2]*d2[2], c = d2[0]**2+d2[1]**2+d2[2]**2;
  const dd = d1[0]*w[0]+d1[1]*w[1]+d1[2]*w[2], e = d2[0]*w[0]+d2[1]*w[1]+d2[2]*w[2];
  const s = (b*e - c*dd) / (a*c - b*b), t = (a*e - b*dd) / (a*c - b*b);
  const p1 = [o1[0]+s*d1[0], o1[1]+s*d1[1], o1[2]+s*d1[2]];
  const p2 = [o2[0]+t*d2[0], o2[1]+t*d2[1], o2[2]+t*d2[2]];
  const mid = [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2];
  const gap = Math.hypot(p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]);
  return { mid, gap, s, t };
}

const [cmd, ...a] = process.argv.slice(2);
if (cmd === 'proj') {
  const cam = byName[a[0]];
  const X = [+a[1], +a[2], +a[3]];
  const [px, py, depth] = project(cam, X);
  console.log(`${a[0]}: px=${px.toFixed(0)} py=${py.toFixed(0)} depth=${depth.toFixed(2)}`);
} else if (cmd === 'projall') {
  const X = [+a[0], +a[1], +a[2]];
  for (const c of cams) {
    const [px, py, depth] = project(c, X);
    if (depth > 0 && px >= 0 && px <= c.width && py >= 0 && py <= c.height)
      console.log(`${c.name.replace('.JPG','')}: px=${px.toFixed(0)} py=${py.toFixed(0)} depth=${depth.toFixed(2)}`);
  }
} else if (cmd === 'tri') {
  const r1 = ray(byName[a[0]], +a[1], +a[2]);
  const r2 = ray(byName[a[3]], +a[4], +a[5]);
  const r = triangulate(r1, r2);
  console.log(`point=[${r.mid.map(v => v.toFixed(3)).join(', ')}] gap=${r.gap.toFixed(3)}`);
} else if (cmd === 'floorpx') {
  // intersect ray with floor y=h (default 0): floorpx VIEW px py [h]
  const h = a[3] !== undefined ? +a[3] : 0;
  const r = ray(byName[a[0]], +a[1], +a[2]);
  const s = (h - r.o[1]) / r.d[1];
  const X = [r.o[0] + s * r.d[0], h, r.o[2] + s * r.d[2]];
  console.log(`floor point=[${X.map(v => v.toFixed(3)).join(', ')}]`);
} else if (cmd === 'wallz' || cmd === 'wallx') {
  // intersect ray with plane z=c (wallz) or x=c (wallx)
  const c = +a[3];
  const r = ray(byName[a[0]], +a[1], +a[2]);
  const i = cmd === 'wallz' ? 2 : 0;
  const s = (c - r.o[i]) / r.d[i];
  const X = [r.o[0] + s * r.d[0], r.o[1] + s * r.d[1], r.o[2] + s * r.d[2]];
  console.log(`point=[${X.map(v => v.toFixed(3)).join(', ')}]`);
}
