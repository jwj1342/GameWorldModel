import fs from 'fs';
import zlib from 'zlib';

const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const cams = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;
const xyz = d.xyz, rgb = d.rgb;

function writePNG(path, w, h, pix) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0;
    pix.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4);
  }
  const idat = zlib.deflateSync(raw);
  const chunks = [];
  function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const t = Buffer.from(type);
    const crcBuf = Buffer.concat([t, data]);
    let crc = 0xFFFFFFFF;
    for (const b of crcBuf) { crc ^= b; for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0); }
    const crcV = Buffer.alloc(4); crcV.writeUInt32BE((crc ^ 0xFFFFFFFF) >>> 0);
    chunks.push(len, t, data, crcV);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; ihdr[9] = 6;
  chunk('IHDR', ihdr); chunk('IDAT', idat); chunk('IEND', Buffer.alloc(0));
  fs.writeFileSync(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), ...chunks]));
}

function quatMat(q) {
  const [x, y, z, w] = q;
  return [
    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
  ];
}

const names = process.argv.slice(2);
for (const c of cams) {
  const short = c.name.replace('.JPG', '');
  if (names.length && !names.includes(short)) continue;
  const R = quatMat(c.quat); // world-from-camera rotation
  // camera-from-world = R^T * (p - pos)
  const W = 779, H = 519;
  const pix = Buffer.alloc(W * H * 4);
  const f = (H / 2) / Math.tan(c.fov_deg * Math.PI / 360);
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    const dx = p[0] - c.pos[0], dy = p[1] - c.pos[1], dz = p[2] - c.pos[2];
    const cx = R[0] * dx + R[3] * dy + R[6] * dz;
    const cy = R[1] * dx + R[4] * dy + R[7] * dz;
    const cz = R[2] * dx + R[5] * dy + R[8] * dz;
    if (cz > -0.05) continue; // behind camera (camera looks -z)
    const u = Math.round(W / 2 + f * cx / -cz);
    const v = Math.round(H / 2 - f * cy / -cz);
    if (u < 0 || u >= W || v < 0 || v >= H) continue;
    // splat 2x2
    for (let a = 0; a < 2; a++) for (let b = 0; b < 2; b++) {
      const uu = u + a, vv = v + b;
      if (uu >= W || vv >= H) continue;
      const idx = (vv * W + uu) * 4;
      pix[idx] = Math.min(255, rgb[i][0] * 2.2);
      pix[idx + 1] = Math.min(255, rgb[i][1] * 2.2);
      pix[idx + 2] = Math.min(255, rgb[i][2] * 2.2);
      pix[idx + 3] = 255;
    }
  }
  writePNG('scratch/proj_' + short + '.png', W, H, pix);
  console.log('wrote scratch/proj_' + short + '.png');
}
