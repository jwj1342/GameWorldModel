// usage: node scratch/proj.mjs VIEW x y z [x y z ...] -> pixel coords
// or:    node scratch/proj.mjs VIEW floor u v [u v ...] -> intersect pixel ray with y=0 (or y=H via "plane H u v")
import { cam, project, ray } from './lib.mjs';
const [name, ...rest] = process.argv.slice(2);
const c = cam(name);
if (rest[0] === 'floor' || rest[0] === 'plane') {
  let H = 0, k = 1;
  if (rest[0] === 'plane') { H = Number(rest[1]); k = 2; }
  for (; k + 1 < rest.length; k += 2) {
    const u = Number(rest[k]), v = Number(rest[k + 1]);
    const d = ray(c, u, v);
    const t = (H - c.pos[1]) / d[1];
    console.log(`(${u},${v}) -> y=${H}: x=${(c.pos[0] + t * d[0]).toFixed(2)} z=${(c.pos[2] + t * d[2]).toFixed(2)} t=${t.toFixed(2)}`);
  }
} else if (rest[0] === 'xplane' || rest[0] === 'zplane') {
  const ax = rest[0] === 'xplane' ? 0 : 2; const H = Number(rest[1]);
  for (let k = 2; k + 1 < rest.length; k += 2) {
    const u = Number(rest[k]), v = Number(rest[k + 1]);
    const d = ray(c, u, v);
    const t = (H - c.pos[ax]) / d[ax];
    console.log(`(${u},${v}) -> ${rest[0]}=${H}: x=${(c.pos[0] + t * d[0]).toFixed(2)} y=${(c.pos[1] + t * d[1]).toFixed(2)} z=${(c.pos[2] + t * d[2]).toFixed(2)} t=${t.toFixed(2)}`);
  }
} else {
  const n = rest.map(Number);
  for (let k = 0; k + 2 < n.length; k += 3) {
    const [u, v, d] = project(c, [n[k], n[k + 1], n[k + 2]]);
    console.log(`(${n[k]},${n[k + 1]},${n[k + 2]}) -> u=${u.toFixed(0)} v=${v.toFixed(0)} depth=${d.toFixed(2)}`);
  }
}
