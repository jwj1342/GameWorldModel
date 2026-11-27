import fs from 'node:fs';
export const P = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
export const C = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;
export function cam(name) {
  const c = C.find((v) => v.name.startsWith(name));
  if (!c) throw new Error('no cam ' + name);
  return c;
}
function qconj(q) { return [-q[0], -q[1], -q[2], q[3]]; }
function qrot(q, v) {
  const [x, y, z, w] = q;
  const [vx, vy, vz] = v;
  const tx = 2 * (y * vz - z * vy), ty = 2 * (z * vx - x * vz), tz = 2 * (x * vy - y * vx);
  return [vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz), vz + w * tz + (x * ty - y * tx)];
}
// world -> pixel (u right, v down) ; returns [u,v,depth]
export function project(c, p) {
  const d = [p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]];
  const l = qrot(qconj(c.quat), d);
  const depth = -l[2];
  const f = 1 / Math.tan((c.fov_deg * Math.PI) / 360);
  const ndcx = (l[0] / depth) * f / c.aspect;
  const ndcy = (l[1] / depth) * f;
  return [(ndcx * 0.5 + 0.5) * c.width, (1 - (ndcy * 0.5 + 0.5)) * c.height, depth];
}
// pixel -> world ray dir
export function ray(c, u, v) {
  const f = 1 / Math.tan((c.fov_deg * Math.PI) / 360);
  const ndcx = (u / c.width) * 2 - 1, ndcy = 1 - (v / c.height) * 2;
  const l = [ndcx * c.aspect / f, ndcy / f, -1];
  const d = qrot(c.quat, l);
  const n = Math.hypot(...d);
  return d.map((x) => x / n);
}
export function pct(arr, q) {
  const a = [...arr].sort((x, y) => x - y);
  if (!a.length) return NaN;
  return a[Math.min(a.length - 1, Math.max(0, Math.round(q * (a.length - 1))))];
}
export function stats(idx) {
  const out = { n: idx.length };
  if (!idx.length) return out;
  for (const [k, ax] of [['x', 0], ['y', 1], ['z', 2]]) {
    const vals = idx.map((i) => P.xyz[i][ax]);
    out[k] = [pct(vals, 0.02), pct(vals, 0.1), pct(vals, 0.5), pct(vals, 0.9), pct(vals, 0.98)].map((v) => +v.toFixed(2));
  }
  const m = [0, 0, 0];
  for (const i of idx) for (let k = 0; k < 3; k++) m[k] += P.rgb[i][k];
  out.rgb = m.map((v) => Math.round(v / idx.length));
  out.hex = '#' + out.rgb.map((v) => v.toString(16).padStart(2, '0')).join('');
  return out;
}
