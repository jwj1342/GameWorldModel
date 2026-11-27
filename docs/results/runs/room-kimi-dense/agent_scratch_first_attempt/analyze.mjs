// Point-cloud analysis helper for scene building.
import fs from 'fs';
const p = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const P = p.xyz; // array of [x,y,z]
const C = p.rgb; // array of [r,g,b]
const N = P.length;

const mode = process.argv[2] || 'grid';
if (mode === 'grid') {
  const cell = parseFloat(process.argv[3] || '0.25');
  const ymin = parseFloat(process.argv[4] || '0.05');
  const ymax = parseFloat(process.argv[5] || '1.2');
  const counts = new Map();
  let minX = 1e9, maxX = -1e9, minZ = 1e9, maxZ = -1e9;
  for (let i = 0; i < N; i++) {
    const [x, y, z] = P[i];
    if (y < ymin || y > ymax) continue;
    const cx = Math.floor(x / cell), cz = Math.floor(z / cell);
    const key = cx + ',' + cz;
    counts.set(key, (counts.get(key) || 0) + 1);
    minX = Math.min(minX, cx); maxX = Math.max(maxX, cx);
    minZ = Math.min(minZ, cz); maxZ = Math.max(maxZ, cz);
  }
  for (let cz = maxZ; cz >= minZ; cz--) {
    let row = String((cz * cell).toFixed(2)).padStart(6) + ' ';
    for (let cx = minX; cx <= maxX; cx++) {
      const c = counts.get(cx + ',' + cz) || 0;
      row += c === 0 ? '  .' : (c < 20 ? '  :' : (c < 80 ? '  o' : (c < 300 ? '  O' : '  #')));
    }
    console.log(row);
  }
  let row = '       ';
  for (let cx = minX; cx <= maxX; cx++) row += ((cx * cell) % 1 === 0 ? '|' : '-');
  console.log(row);
  console.log('x cols from', (minX * cell).toFixed(2), 'to', ((maxX + 1) * cell).toFixed(2), ' cell=', cell);
} else if (mode === 'box') {
  const [x0, x1, y0, y1, z0, z1] = process.argv.slice(3).map(Number);
  const sel = [];
  for (let i = 0; i < N; i++) {
    const [x, y, z] = P[i];
    if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && z >= z0 && z <= z1) sel.push(i);
  }
  if (sel.length === 0) { console.log('no points'); process.exit(0); }
  const q = (arr, f) => { arr = [...arr].sort((a, b) => a - b); return arr[Math.floor(f * (arr.length - 1))]; };
  const xs = sel.map(i => P[i][0]), ys = sel.map(i => P[i][1]), zs = sel.map(i => P[i][2]);
  console.log('n', sel.length);
  console.log('x', q(xs, 0.01).toFixed(3), '..', q(xs, 0.99).toFixed(3));
  console.log('y', q(ys, 0.01).toFixed(3), '..', q(ys, 0.99).toFixed(3));
  console.log('z', q(zs, 0.01).toFixed(3), '..', q(zs, 0.99).toFixed(3));
  let r = 0, g = 0, b = 0;
  for (const i of sel) { r += C[i][0]; g += C[i][1]; b += C[i][2]; }
  console.log('avg rgb', Math.round(r / sel.length), Math.round(g / sel.length), Math.round(b / sel.length));
}
