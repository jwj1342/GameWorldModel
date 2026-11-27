// usage: node scratch/box.mjs xmin xmax ymin ymax zmin zmax  -> stats of points in box
import { P, stats } from './lib.mjs';
const a = process.argv.slice(2).map(Number);
for (let k = 0; k + 5 < a.length; k += 6) {
  const [x0, x1, y0, y1, z0, z1] = a.slice(k, k + 6);
  const idx = [];
  P.xyz.forEach((p, i) => { if (p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1 && p[2] >= z0 && p[2] <= z1) idx.push(i); });
  console.log(JSON.stringify(stats(idx)));
}
