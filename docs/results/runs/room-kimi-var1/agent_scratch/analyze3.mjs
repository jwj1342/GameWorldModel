// 分区统计点云：沿东墙/西墙的 z 分布、各区域包围盒
import { readFileSync } from 'node:fs';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));

function box(name, pred) {
  const sel = [];
  for (let i = 0; i < xyz.length; i++) if (pred(xyz[i], rgb[i])) sel.push(i);
  if (!sel.length) { console.log(name, ': none'); return []; }
  const mn = [1e9, 1e9, 1e9], mx = [-1e9, -1e9, -1e9];
  for (const i of sel) for (let k = 0; k < 3; k++) {
    mn[k] = Math.min(mn[k], xyz[i][k]); mx[k] = Math.max(mx[k], xyz[i][k]);
  }
  console.log(name, 'n=' + sel.length, 'min', mn.map(v => v.toFixed(2)).join(','), 'max', mx.map(v => v.toFixed(2)).join(','));
  return sel;
}
const inRoom = p => p[0] > -1.7 && p[0] < 1.7 && p[2] > -2.8 && p[2] < 4.2 && p[1] > -0.3 && p[1] < 3;

// 东墙条带 z 直方图（每 0.2m 一格，报告高度范围）
console.log('--- east strip x>0.85, z histogram ---');
for (let z0 = -2.6; z0 < 4.0; z0 += 0.2) {
  const sel = box._ = [];
  let n = 0, ymax = -9, ymin = 9;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!inRoom(p) || p[0] < 0.85 || p[2] < z0 || p[2] >= z0 + 0.2) continue;
    n++; ymax = Math.max(ymax, p[1]); ymin = Math.min(ymin, p[1]);
  }
  if (n > 3) console.log(z0.toFixed(1), 'n=' + n, 'y', ymin.toFixed(2), '..', ymax.toFixed(2));
}
console.log('--- west strip x<-0.75, z histogram ---');
for (let z0 = -2.6; z0 < 4.0; z0 += 0.2) {
  let n = 0, ymax = -9, ymin = 9;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!inRoom(p) || p[0] > -0.75 || p[2] < z0 || p[2] >= z0 + 0.2) continue;
    n++; ymax = Math.max(ymax, p[1]); ymin = Math.min(ymin, p[1]);
  }
  if (n > 3) console.log(z0.toFixed(1), 'n=' + n, 'y', ymin.toFixed(2), '..', ymax.toFixed(2));
}
console.log('--- south strip z<-1.9, x histogram ---');
for (let x0 = -1.6; x0 < 1.6; x0 += 0.2) {
  let n = 0, ymax = -9, ymin = 9;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!inRoom(p) || p[2] > -1.9 || p[0] < x0 || p[0] >= x0 + 0.2) continue;
    n++; ymax = Math.max(ymax, p[1]); ymin = Math.min(ymin, p[1]);
  }
  if (n > 3) console.log(x0.toFixed(1), 'n=' + n, 'y', ymin.toFixed(2), '..', ymax.toFixed(2));
}
console.log('--- north strip z>2.2, x histogram ---');
for (let x0 = -1.6; x0 < 1.6; x0 += 0.2) {
  let n = 0, ymax = -9, ymin = 9;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (!inRoom(p) || p[2] < 2.2 || p[0] < x0 || p[0] >= x0 + 0.2) continue;
    n++; ymax = Math.max(ymax, p[1]); ymin = Math.min(ymin, p[1]);
  }
  if (n > 3) console.log(x0.toFixed(1), 'n=' + n, 'y', ymin.toFixed(2), '..', ymax.toFixed(2));
}
