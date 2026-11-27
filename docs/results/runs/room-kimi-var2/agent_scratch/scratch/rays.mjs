import fs from 'fs';
const cams = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;

function quatMat(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

// ray through normalized pixel (u,v), u right, v down, 0..1
function ray(camName, u, v) {
  const c = cams.find(c => c.name.includes(camName));
  const R = quatMat(c.quat);
  const f = 0.5 / Math.tan(c.fov_deg * Math.PI / 360); // in v units
  const aspect = c.aspect;
  const x = (u - 0.5) * aspect / f * 0.5 * 2; // careful below
  // simpler: tan half-fov vertical = t; y = (0.5 - v)*2*t ; x = (u-0.5)*2*t*aspect
  const t = Math.tan(c.fov_deg * Math.PI / 360);
  const cy = (0.5 - v) * 2 * t;
  const cx = (u - 0.5) * 2 * t * aspect;
  const cz = -1;
  // world dir = R * (cx,cy,cz)
  const dxx = R[0] * cx + R[1] * cy + R[2] * cz;
  const dyy = R[3] * cx + R[4] * cy + R[5] * cz;
  const dzz = R[6] * cx + R[7] * cy + R[8] * cz;
  const n = Math.hypot(dxx, dyy, dzz);
  return { o: c.pos, d: [dxx / n, dyy / n, dzz / n] };
}

// closest point between two rays
function intersect(r1, r2) {
  const [o1, d1] = [r1.o, r1.d], [o2, d2] = [r2.o, r2.d];
  const w = [o1[0] - o2[0], o1[1] - o2[1], o1[2] - o2[2]];
  const a = d1[0] ** 2 + d1[1] ** 2 + d1[2] ** 2;
  const b = d1[0] * d2[0] + d1[1] * d2[1] + d1[2] * d2[2];
  const c = d2[0] ** 2 + d2[1] ** 2 + d2[2] ** 2;
  const d = d1[0] * w[0] + d1[1] * w[1] + d1[2] * w[2];
  const e = d2[0] * w[0] + d2[1] * w[1] + d2[2] * w[2];
  const den = a * c - b * b;
  const s = (b * e - c * d) / den;
  const t = (a * e - b * d) / den;
  const p1 = [o1[0] + d1[0] * s, o1[1] + d1[1] * s, o1[2] + d1[2] * s];
  const p2 = [o2[0] + d2[0] * t, o2[1] + d2[1] * t, o2[2] + d2[2] * t];
  return { p: [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2, (p1[2] + p2[2]) / 2], gap: Math.hypot(p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]) };
}

const args = process.argv.slice(2);
// usage: node rays.mjs CAM u v CAM u v  -> intersect; or CAM u v -> print ray + floor hit
if (args.length === 3) {
  const r = ray(args[0], +args[1], +args[2]);
  console.log('origin', r.o.map(x => x.toFixed(2)).join(','), 'dir', r.d.map(x => x.toFixed(3)).join(','));
  if (r.d[1] < -0.01) {
    const t = -r.o[1] / r.d[1];
    console.log('floor hit:', (r.o[0] + r.d[0] * t).toFixed(2), (r.o[2] + r.d[2] * t).toFixed(2), 'dist', t.toFixed(2));
  }
} else if (args.length === 6) {
  const r1 = ray(args[0], +args[1], +args[2]);
  const r2 = ray(args[3], +args[4], +args[5]);
  const res = intersect(r1, r2);
  console.log('point', res.p.map(x => x.toFixed(2)).join(', '), 'gap', res.gap.toFixed(3));
}
