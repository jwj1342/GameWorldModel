import fs from 'fs';
import zlib from 'zlib';

const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const cams = JSON.parse(fs.readFileSync('inputs/cameras.json', 'utf8')).views;
const xyz = d.xyz, rgb = d.rgb;

// minimal PNG writer
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

// camera info: position + forward vector
for (const c of cams) {
  const R = quatMat(c.quat);
  const fwd = [-R[2], -R[5], -R[8]];
  const right = [R[0], R[3], R[6]];
  console.log(c.name, 'pos', c.pos.map(v => v.toFixed(2)).join(','),
    'fwd', fwd.map(v => v.toFixed(2)).join(','),
    'right', right.map(v => v.toFixed(2)).join(','));
}

// Top-down orthographic render, color = actual rgb, with camera markers
const W = 800, H = 1200;
const x0 = -2.0, z0 = -3.0, x1 = 2.0, z1 = 4.5;
const pix = Buffer.alloc(W * H * 4);
for (let i = 0; i < W * H; i++) { pix[i * 4] = 24; pix[i * 4 + 1] = 24; pix[i * 4 + 2] = 24; pix[i * 4 + 3] = 255; }
const zbuf = new Float32Array(W * H).fill(-1e9);
function plot(wx, wz, y, r, g, b) {
  const px = Math.round((wx - x0) / (x1 - x0) * W);
  const pz = Math.round((wz - z0) / (z1 - z0) * H);
  if (px < 0 || px >= W || pz < 0 || pz >= H) return;
  const idx = pz * W + px;
  if (y > zbuf[idx]) {
    zbuf[idx] = y;
    pix[idx * 4] = r; pix[idx * 4 + 1] = g; pix[idx * 4 + 2] = b;
  }
}
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i];
  if (p[1] < -0.1 || p[1] > 2.5) continue;
  plot(p[0], p[2], p[1], rgb[i][0], rgb[i][1], rgb[i][2]);
}
// cameras as red dots with a line showing fwd
for (const c of cams) {
  const R = quatMat(c.quat);
  const fwd = [-R[2], -R[5], -R[8]];
  for (let t = 0; t < 40; t++) {
    plot(c.pos[0] + fwd[0] * t * 0.01, c.pos[2] + fwd[2] * t * 0.01, 5, 255, 0, 0);
  }
  plot(c.pos[0], c.pos[2], 6, 255, 255, 0);
}
writePNG('scratch/topdown.png', W, H, pix);
console.log('wrote scratch/topdown.png');
