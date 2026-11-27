// top-down ASCII density map. usage: node scratch/map.mjs ymin ymax [cell] [xmin xmax zmin zmax]
import { P } from './lib.mjs';
const a = process.argv.slice(2).map(Number);
const [ymin, ymax] = a;
const cell = a[2] || 0.1;
const xmin = a[3] ?? -2.2, xmax = a[4] ?? 2.2, zmin = a[5] ?? -3.4, zmax = a[6] ?? 4.6;
const nx = Math.round((xmax - xmin) / cell), nz = Math.round((zmax - zmin) / cell);
const g = Array.from({ length: nz }, () => new Array(nx).fill(0));
for (const p of P.xyz) {
  if (p[1] < ymin || p[1] > ymax) continue;
  const i = Math.floor((p[0] - xmin) / cell), j = Math.floor((p[2] - zmin) / cell);
  if (i < 0 || i >= nx || j < 0 || j >= nz) continue;
  g[j][i]++;
}
const chars = ' .:-=+*#%@';
let head = '       ';
for (let i = 0; i < nx; i++) { const x = xmin + i * cell; head += (Math.abs(Math.round(x * 10) % 5) === 0 && Math.abs(x*10 - Math.round(x*10)) < 1e-6) ? '|' : ' '; }
console.log(head + '  x from ' + xmin + ' to ' + xmax + ' (| every 0.5)');
for (let j = 0; j < nz; j++) {
  const z = zmin + j * cell;
  let row = (z >= 0 ? ' ' : '') + z.toFixed(2) + '  ';
  row = row.padStart(7);
  for (let i = 0; i < nx; i++) { const v = g[j][i]; row += chars[Math.min(chars.length - 1, v === 0 ? 0 : 1 + Math.floor(Math.log2(v)))]; }
  console.log(row);
}
