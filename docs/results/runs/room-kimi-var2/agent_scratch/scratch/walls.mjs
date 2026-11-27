import fs from 'fs';
const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const xyz = d.xyz, rgb = d.rgb;

// +x wall profile: for each z slice, max y and mean color of points with x > 0.85
console.log('=== +x wall (x>0.85): z-slice -> count, maxY, mean rgb ===');
for (let z = -2.6; z < 2.6; z += 0.2) {
  let n = 0, maxy = 0, sr = 0, sg = 0, sb = 0, minx = 9;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[0] < 0.85 || p[2] < z || p[2] >= z + 0.2 || p[1] < 0.1) continue;
    n++; maxy = Math.max(maxy, p[1]); minx = Math.min(minx, p[0]);
    sr += rgb[i][0]; sg += rgb[i][1]; sb += rgb[i][2];
  }
  if (n < 3) { console.log('z ' + z.toFixed(1) + ': n=' + n); continue; }
  console.log('z ' + z.toFixed(1).padStart(5) + ': n=' + String(n).padStart(4) + ' maxY=' + maxy.toFixed(2) + ' minX=' + minx.toFixed(2) + ' rgb(' + Math.round(sr / n) + ',' + Math.round(sg / n) + ',' + Math.round(sb / n) + ')');
}
console.log('=== -x wall (x<-1.2): z-slice ===');
for (let z = -2.6; z < 4.2; z += 0.2) {
  let n = 0, maxy = 0, sr = 0, sg = 0, sb = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[0] > -1.2 || p[2] < z || p[2] >= z + 0.2 || p[1] < 0.1) continue;
    n++; maxy = Math.max(maxy, p[1]);
    sr += rgb[i][0]; sg += rgb[i][1]; sb += rgb[i][2];
  }
  if (n < 3) { console.log('z ' + z.toFixed(1) + ': n=' + n); continue; }
  console.log('z ' + z.toFixed(1).padStart(5) + ': n=' + String(n).padStart(4) + ' maxY=' + maxy.toFixed(2) + ' rgb(' + Math.round(sr / n) + ',' + Math.round(sg / n) + ',' + Math.round(sb / n) + ')');
}
console.log('=== -z wall (z<-2.2): x-slice ===');
for (let x = -1.6; x < 1.6; x += 0.2) {
  let n = 0, maxy = 0, sr = 0, sg = 0, sb = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[2] > -2.2 || p[0] < x || p[0] >= x + 0.2 || p[1] < 0.1) continue;
    n++; maxy = Math.max(maxy, p[1]);
    sr += rgb[i][0]; sg += rgb[i][1]; sb += rgb[i][2];
  }
  if (n < 3) { console.log('x ' + x.toFixed(1) + ': n=' + n); continue; }
  console.log('x ' + x.toFixed(1).padStart(5) + ': n=' + String(n).padStart(4) + ' maxY=' + maxy.toFixed(2) + ' rgb(' + Math.round(sr / n) + ',' + Math.round(sg / n) + ',' + Math.round(sb / n) + ')');
}
console.log('=== far end (z>2.9): x-slice ===');
for (let x = -1.6; x < 1.6; x += 0.2) {
  let n = 0, maxy = 0, sr = 0, sg = 0, sb = 0;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[2] < 2.9 || p[0] < x || p[0] >= x + 0.2 || p[1] < 0.05) continue;
    n++; maxy = Math.max(maxy, p[1]);
    sr += rgb[i][0]; sg += rgb[i][1]; sb += rgb[i][2];
  }
  if (n < 3) { console.log('x ' + x.toFixed(1) + ': n=' + n); continue; }
  console.log('x ' + x.toFixed(1).padStart(5) + ': n=' + String(n).padStart(4) + ' maxY=' + maxy.toFixed(2) + ' rgb(' + Math.round(sr / n) + ',' + Math.round(sg / n) + ',' + Math.round(sb / n) + ')');
}
