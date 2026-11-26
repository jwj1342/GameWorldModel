import fs from 'fs';
import zlib from 'zlib';

const d = JSON.parse(fs.readFileSync('inputs/points.json', 'utf8'));
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

const x0 = -1.8, z0 = -2.8, x1 = 1.8, z1 = 4.3;
const W = 720, H = 1420;
function mkpix() {
  const pix = Buffer.alloc(W * H * 4);
  for (let i = 0; i < W * H; i++) { pix[i * 4 + 3] = 255; }
  return pix;
}
function plotter(pix, zbuf, brighten) {
  return (wx, wz, y, r, g, b) => {
    const px = Math.round((wx - x0) / (x1 - x0) * W);
    const pz = Math.round((wz - z0) / (z1 - z0) * H);
    if (px < 0 || px >= W || pz < 0 || pz >= H) return;
    const idx = pz * W + px;
    if (y > zbuf[idx]) {
      zbuf[idx] = y;
      pix[idx * 4] = Math.min(255, r * brighten);
      pix[idx * 4 + 1] = Math.min(255, g * brighten);
      pix[idx * 4 + 2] = Math.min(255, b * brighten);
    }
  };
}

// 1: floor band only (y < 0.12), true color brightened -> see rugs
{
  const pix = mkpix(); const zb = new Float32Array(W * H).fill(-1e9);
  const plot = plotter(pix, zb, 2.6);
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[1] < -0.05 || p[1] > 0.12) continue;
    plot(p[0], p[2], p[1], rgb[i][0], rgb[i][1], rgb[i][2]);
  }
  writePNG('scratch/floor.png', W, H, pix);
}
// 2: height-coded, 0..1.2m
{
  const pix = mkpix(); const zb = new Float32Array(W * H).fill(-1e9);
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[1] < -0.05 || p[1] > 2.2) continue;
    const t = Math.max(0, Math.min(1, p[1] / 1.4));
    // blue(low) -> green -> red(high)
    const r = Math.round(255 * Math.min(1, Math.max(0, t * 2 - 0.6)));
    const g = Math.round(255 * (1 - Math.abs(t - 0.5) * 2));
    const b = Math.round(255 * Math.min(1, Math.max(0, 1.2 - t * 2)));
    const px = Math.round((p[0] - x0) / (x1 - x0) * W);
    const pz = Math.round((p[2] - z0) / (z1 - z0) * H);
    if (px < 0 || px >= W || pz < 0 || pz >= H) continue;
    const idx = pz * W + px;
    if (p[1] > zb[idx]) { zb[idx] = p[1]; pix[idx * 4] = r; pix[idx * 4 + 1] = g; pix[idx * 4 + 2] = b; pix[idx * 4 + 3] = 255; }
  }
  writePNG('scratch/height.png', W, H, pix);
}
// 3: side elevation looking from +x (see z horizontal, y vertical)
{
  const W2 = 1420, H2 = 500;
  const pix = Buffer.alloc(W2 * H2 * 4);
  for (let i = 0; i < W2 * H2; i++) { pix[i * 4 + 3] = 255; }
  const zb = new Float32Array(W2 * H2).fill(-1e9);
  const y0 = -0.2, y1 = 2.3;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[2] < z0 || p[2] > z1 || p[1] < y0 || p[1] > y1) continue;
    const px = Math.round((p[2] - z0) / (z1 - z0) * W2);
    const py = Math.round((1 - (p[1] - y0) / (y1 - y0)) * H2);
    if (px < 0 || px >= W2 || py < 0 || py >= H2) continue;
    const idx = py * W2 + px;
    if (p[0] > zb[idx]) {
      zb[idx] = p[0];
      pix[idx * 4] = Math.min(255, rgb[i][0] * 2.6);
      pix[idx * 4 + 1] = Math.min(255, rgb[i][1] * 2.6);
      pix[idx * 4 + 2] = Math.min(255, rgb[i][2] * 2.6);
    }
  }
  writePNG('scratch/elev_from_xpos.png', W2, H2, pix);
}
// 4: side elevation looking from -x
{
  const W2 = 1420, H2 = 500;
  const pix = Buffer.alloc(W2 * H2 * 4);
  for (let i = 0; i < W2 * H2; i++) { pix[i * 4 + 3] = 255; }
  const zb = new Float32Array(W2 * H2).fill(1e9);
  const y0 = -0.2, y1 = 2.3;
  for (let i = 0; i < xyz.length; i++) {
    const p = xyz[i];
    if (p[2] < z0 || p[2] > z1 || p[1] < y0 || p[1] > y1) continue;
    const px = Math.round((p[2] - z0) / (z1 - z0) * W2);
    const py = Math.round((1 - (p[1] - y0) / (y1 - y0)) * H2);
    if (px < 0 || px >= W2 || py < 0 || py >= H2) continue;
    const idx = py * W2 + px;
    if (p[0] < zb[idx]) {
      zb[idx] = p[0];
      pix[idx * 4] = Math.min(255, rgb[i][0] * 2.6);
      pix[idx * 4 + 1] = Math.min(255, rgb[i][1] * 2.6);
      pix[idx * 4 + 2] = Math.min(255, rgb[i][2] * 2.6);
    }
  }
  writePNG('scratch/elev_from_xneg.png', W2, H2, pix);
}
console.log('done');
