// usage: node scratch/grid.mjs VIEW [cellpx=60]  -> per-cell median xyz of visible sparse points
import { P, cam, project, pct } from './lib.mjs';
const [name, cellArg] = process.argv.slice(2);
const c = cam(name);
const cell = Number(cellArg) || 60;
const nx = Math.ceil(c.width / cell), ny = Math.ceil(c.height / cell);
const g = Array.from({ length: ny }, () => Array.from({ length: nx }, () => []));
P.xyz.forEach((p, i) => {
  const [u, v, d] = project(c, p);
  if (d <= 0.05 || u < 0 || v < 0 || u >= c.width || v >= c.height) return;
  g[Math.floor(v / cell)][Math.floor(u / cell)].push([i, d]);
});
let head = 'v\\u   ';
for (let i = 0; i < nx; i++) head += String(i * cell + cell / 2).padEnd(17);
console.log(head);
for (let j = 0; j < ny; j++) {
  let row = String(j * cell + cell / 2).padEnd(6);
  for (let i = 0; i < nx; i++) {
    const h = g[j][i];
    if (h.length < 2) { row += '.'.padEnd(17); continue; }
    const xs = h.map((e) => P.xyz[e[0]][0]), ys = h.map((e) => P.xyz[e[0]][1]), zs = h.map((e) => P.xyz[e[0]][2]);
    row += `${pct(xs, 0.5).toFixed(1)},${pct(ys, 0.5).toFixed(1)},${pct(zs, 0.5).toFixed(1)}(${h.length})`.padEnd(17);
  }
  console.log(row);
}
