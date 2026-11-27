// 把点云投影到指定相机，输出 PPM（再用 sips 转 png），与照片对比
import { readFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
const { xyz, rgb } = JSON.parse(readFileSync('inputs/points.json', 'utf8'));
const cams = JSON.parse(readFileSync('inputs/cameras.json', 'utf8')).views;

function quatToMat(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}
const name = process.argv[2] || 'DSCF4721';
const c = cams.find(v => v.name.startsWith(name));
const m = quatToMat(c.quat);
// world->camera: R^T * (p - pos)
const W = 779, H = 519;
const img = new Uint8Array(W * H * 3);
const f = 0.5 * H / Math.tan(c.fov_deg * Math.PI / 360);
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i];
  const d = [p[0] - c.pos[0], p[1] - c.pos[1], p[2] - c.pos[2]];
  const cx = m[0] * d[0] + m[3] * d[1] + m[6] * d[2];
  const cy = m[1] * d[0] + m[4] * d[1] + m[7] * d[2];
  const cz = m[2] * d[0] + m[5] * d[1] + m[8] * d[2];
  if (cz > -0.05) continue;
  const uu = Math.round(W / 2 + f * cx / -cz);
  const vv = Math.round(H / 2 - f * cy / -cz);
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    const px = uu + dx, py = vv + dy;
    if (px < 0 || px >= W || py < 0 || py >= H) continue;
    const o = (py * W + px) * 3;
    img[o] = Math.min(255, rgb[i][0] + 60); img[o + 1] = Math.min(255, rgb[i][1] + 60); img[o + 2] = Math.min(255, rgb[i][2] + 60);
  }
}
writeFileSync(`preview_pts_${name}.ppm`, Buffer.concat([Buffer.from(`P6\n${W} ${H}\n255\n`), Buffer.from(img)]));
execSync(`sips -s format png preview_pts_${name}.ppm --out preview_pts_${name}.png`, { stdio: 'inherit' });
console.log('wrote preview_pts_' + name + '.png');
