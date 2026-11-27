// usage: node scratch/tri.mjs label VIEW u v VIEW u v ... [-- label VIEW u v ...]
// least-squares 3D point closest to all pixel rays; prints residual per ray (m)
import { cam, ray, project } from './lib.mjs';
function solve3(A, b) {
  const m = A.map((r, i) => [...r, b[i]]);
  for (let i = 0; i < 3; i++) {
    let p = i;
    for (let r = i + 1; r < 3; r++) if (Math.abs(m[r][i]) > Math.abs(m[p][i])) p = r;
    [m[i], m[p]] = [m[p], m[i]];
    for (let r = 0; r < 3; r++) if (r !== i) { const f = m[r][i] / m[i][i]; for (let c = i; c < 4; c++) m[r][c] -= f * m[i][c]; }
  }
  return [m[0][3] / m[0][0], m[1][3] / m[1][1], m[2][3] / m[2][2]];
}
function tri(label, obs) {
  const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], b = [0, 0, 0];
  const rays = obs.map(([n, u, v]) => { const c = cam(n); return { c, d: ray(c, u, v), u, v, n }; });
  for (const { c, d } of rays) {
    for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
      const m = (i === j ? 1 : 0) - d[i] * d[j];
      A[i][j] += m; b[i] += m * c.pos[j];
    }
  }
  const X = solve3(A, b);
  const res = rays.map(({ c, u, v, n }) => { const [pu, pv] = project(c, X); return `${n.slice(4)}:${Math.hypot(pu - u, pv - v).toFixed(0)}px`; });
  console.log(`${label}: x=${X[0].toFixed(2)} y=${X[1].toFixed(2)} z=${X[2].toFixed(2)}   reproj ${res.join(' ')}`);
}
const args = process.argv.slice(2);
let cur = null, obs = [];
let i = 0;
while (i < args.length) {
  if (args[i] === '--') { i++; continue; }
  if (!args[i].startsWith('DSCF')) { if (cur) tri(cur, obs); cur = args[i]; obs = []; i++; continue; }
  obs.push([args[i], Number(args[i + 1]), Number(args[i + 2])]); i += 3;
}
if (cur) tri(cur, obs);
