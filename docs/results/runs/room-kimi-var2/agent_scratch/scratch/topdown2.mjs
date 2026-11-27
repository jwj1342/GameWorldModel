import fs from 'fs';
import zlib from 'zlib';
const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
const xyz = d.xyz, rgb = d.rgb;

function writePNG(path, w, h, pix) {
  const raw = Buffer.alloc((w * 4 + 1) * h);
  for (let y = 0; y < h; y++) { raw[y * (w * 4 + 1)] = 0; pix.copy(raw, y * (w * 4 + 1) + 1, y * w * 4, (y + 1) * w * 4); }
  const idat = zlib.deflateSync(raw);
  const chunks = [];
  function chunk(type, data) {
    const len = Buffer.alloc(4); len.writeUInt32BE(data.length);
    const t = Buffer.from(type); const crcBuf = Buffer.concat([t, data]);
    let crc = 0xFFFFFFFF;
    for (const b of crcBuf) { crc ^= b; for (let k = 0; k < 8; k++) crc = (crc >>> 1) ^ (crc & 1 ? 0xEDB88320 : 0); }
    const crcV = Buffer.alloc(4); crcV.writeUInt32BE((crc ^ 0xFFFFFFFF) >>> 0);
    chunks.push(len, t, data, crcV);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0); ihdr.writeUInt32BE(h, 4); ihdr[8] = 8; ihdr[9] = 6;
  chunk('IHDR', ihdr); chunk('IDAT', idat); chunk('IEND', Buffer.alloc(0));
  fs.writeFileSync(path, Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), ...chunks]));
}

const x0 = -1.7, z0 = -2.7, x1 = 1.7, z1 = 4.2;
const scale = 220; // px per meter
const W = Math.round((x1 - x0) * scale), H = Math.round((z1 - z0) * scale);
const zb = new Float32Array(W * H).fill(-1e9);
const col = Buffer.alloc(W * H * 3);
const hgt = Buffer.alloc(W * H * 4);
for (let i = 0; i < xyz.length; i++) {
  const p = xyz[i];
  if (p[1] < -0.1 || p[1] > 2.4) continue;
  const px = Math.round((p[0] - x0) * scale), pz = Math.round((p[2] - z0) * scale);
  if (px < 1 || px >= W - 1 || pz < 1 || pz >= H - 1) continue;
  for (let a = -1; a <= 1; a++) for (let b = -1; b <= 1; b++) {
    const idx = (pz + b) * W + (px + a);
    if (p[1] >= zb[idx] - 0.03) {
      zb[idx] = Math.max(zb[idx], p[1]);
      const g = 1 / (1 + Math.exp(-(p[1] - 0.4) * 3));
      col[idx * 3] = Math.min(255, rgb[i][0] * (1.5 + g));
      col[idx * 3 + 1] = Math.min(255, rgb[i][1] * (1.5 + g));
      col[idx * 3 + 2] = Math.min(255, rgb[i][2] * (1.5 + g));
    }
  }
}
const pixC = Buffer.alloc(W * H * 4), pixH = Buffer.alloc(W * H * 4);
for (let i = 0; i < W * H; i++) {
  pixC[i * 4] = col[i * 3]; pixC[i * 4 + 1] = col[i * 3 + 1]; pixC[i * 4 + 2] = col[i * 3 + 2]; pixC[i * 4 + 3] = 255;
  const y = zb[i];
  let v = 0;
  if (y > -1e8) v = Math.round(30 + 225 * Math.min(1, Math.max(0, (y + 0.05) / 1.6)));
  pixH[i * 4] = v; pixH[i * 4 + 1] = v; pixH[i * 4 + 2] = v; pixH[i * 4 + 3] = 255;
}
// grid every 0.5m in red-ish on color image, blue on height
for (let gx = Math.ceil(x0 * 2) / 2; gx <= x1; gx += 0.5) {
  const px = Math.round((gx - x0) * scale);
  for (let z = 0; z < H; z++) {
    const i = (z * W + px) * 4;
    pixC[i] = 255; pixC[i + 1] = 0; pixC[i + 2] = 0;
    pixH[i] = 0; pixH[i + 1] = 0; pixH[i + 2] = 255;
  }
}
for (let gz = Math.ceil(z0 * 2) / 2; gz <= z1; gz += 0.5) {
  const pz = Math.round((gz - z0) * scale);
  for (let x = 0; x < W; x++) {
    const i = (pz * W + x) * 4;
    pixC[i] = 255; pixC[i + 1] = 0; pixC[i + 2] = 0;
    pixH[i] = 0; pixH[i + 1] = 0; pixH[i + 2] = 255;
  }
}
writePNG('scratch/top_color.png', W, H, pixC);
writePNG('scratch/top_height.png', W, H, pixH);
console.log('W,H =', W, H, 'x0,z0 =', x0, z0, 'scale', scale);
