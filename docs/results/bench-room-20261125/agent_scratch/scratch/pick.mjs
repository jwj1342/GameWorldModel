// usage: node scratch/pick.mjs VIEW u v [radius_px]  (repeat triples: u v r)
// lists points projecting within radius of the pixel, grouped by depth, nearest-depth cluster first
import { P, cam, project, stats, pct } from './lib.mjs';
const [name, ...rest] = process.argv.slice(2);
const c = cam(name);
const nums = rest.map(Number);
for (let k = 0; k + 2 < nums.length + 1; k += 3) {
  const u = nums[k], v = nums[k + 1], r = nums[k + 2] || 15;
  if (u === undefined || v === undefined) break;
  const hits = [];
  P.xyz.forEach((p, i) => {
    const [pu, pv, d] = project(c, p);
    if (d <= 0.05) return;
    if (Math.hypot(pu - u, pv - v) <= r) hits.push([i, d]);
  });
  hits.sort((a, b) => a[1] - b[1]);
  // cluster by depth gaps
  const clusters = [];
  let cur = [];
  for (const h of hits) {
    if (cur.length && h[1] - cur[cur.length - 1][1] > 0.25) { clusters.push(cur); cur = []; }
    cur.push(h);
  }
  if (cur.length) clusters.push(cur);
  console.log(`pixel (${u},${v}) r=${r}: ${hits.length} pts, ${clusters.length} depth clusters`);
  for (const cl of clusters) {
    const s = stats(cl.map((h) => h[0]));
    console.log(`  depth ${cl[0][1].toFixed(2)}-${cl[cl.length - 1][1].toFixed(2)} n=${s.n} x~${s.x[2]} [${s.x[0]},${s.x[4]}] y~${s.y[2]} [${s.y[0]},${s.y[4]}] z~${s.z[2]} [${s.z[0]},${s.z[4]}] ${s.hex}`);
  }
}
