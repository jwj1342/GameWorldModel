import fs from 'fs';
const cams = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;
function rot(q, v) {
  const [x, y, z, w] = q;
  const dot = x * v[0] + y * v[1] + z * v[2];
  const cx = y * v[2] - z * v[1], cy = z * v[0] - x * v[2], cz = x * v[1] - y * v[0];
  const k = w * w - (x * x + y * y + z * z);
  return [
    2 * dot * x + k * v[0] + 2 * w * cx,
    2 * dot * y + k * v[1] + 2 * w * cy,
    2 * dot * z + k * v[2] + 2 * w * cz,
  ];
}
for (const c of cams) {
  const f = rot(c.quat, [0, 0, -1]);
  const yaw = Math.atan2(f[0], -f[2]) * 180 / Math.PI; // 0 = looking -z, 90 = +x
  const pitch = Math.asin(f[1]) * 180 / Math.PI;
  console.log(c.name.replace('.JPG', ''), 'pos', c.pos.map(v => v.toFixed(2)).join(','),
    'fwd', f.map(v => v.toFixed(2)).join(','), 'yaw', yaw.toFixed(0), 'pitch', pitch.toFixed(0));
}
