// record.mjs — 把程序在回放模式下逐帧渲染成 PNG 序列 + 逐帧真值（用于合成调试片段）
// 用法: node record.mjs --game DIR --out DIR [--fps 30] [--duration 8] [--width 960 --height 540]
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs, savePng } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const fps = Number(args.fps ?? 30), duration = Number(args.duration ?? 8);
const width = Number(args.width ?? 960), height = Number(args.height ?? 540);
const outDir = args.out; fs.mkdirSync(path.join(outDir, 'frames'), { recursive: true });
const g = await openGame(args.game, { width, height, logFile: path.join(outDir, 'browser.log') });
await g.page.evaluate(() => window.__game.mode('replay'));
const gt = fs.createWriteStream(path.join(outDir, 'gt_states.jsonl'));
const n = Math.round(duration * fps);
const t0 = Date.now();
for (let i = 0; i < n; i++) {
  const t = i / fps;
  const { b64, state, cam } = await g.page.evaluate((t) => { window.__game.seek(t); return { b64: window.__game.render('rgb'), state: window.__game.state(), cam: window.__game.cameraInfo() }; }, t);
  savePng(b64, path.join(outDir, 'frames', `f_${String(i).padStart(5, '0')}.png`));
  gt.write(JSON.stringify({ i, t, camera: cam, objects: state.objects }) + '\n');
}
gt.end();
fs.writeFileSync(path.join(outDir, 'record.json'), JSON.stringify({ fps, duration, width, height, frames: n, elapsed_ms: Date.now() - t0 }, null, 2));
await g.close();
console.log(`recorded ${n} frames to ${outDir}/frames (${Date.now() - t0} ms). Assemble with: ffmpeg -framerate ${fps} -i ${outDir}/frames/f_%05d.png -c:v libx264 -pix_fmt yuv420p ${outDir}/clip.mp4`);
