// render.mjs — 回放模式下在给定时刻渲染 rgb / depth / id 三个 pass
// 用法: node render.mjs --game DIR --times 0,1.5,3 --out DIR [--width 640 --height 360] [--passes rgb,depth,id]
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs, savePng } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const gameDir = args.game, outDir = args.out;
const times = (args.times ?? '0').split(',').map(Number);
const passes = (args.passes ?? 'rgb,depth,id').split(',');
const width = Number(args.width ?? 640), height = Number(args.height ?? 360);
fs.mkdirSync(outDir, { recursive: true });
const t0 = Date.now();
const g = await openGame(gameDir, { width, height, logFile: path.join(outDir, 'browser.log') });
await g.page.evaluate(() => window.__game.mode('replay'));
const index = { width, height, passes, frames: [], camera_far: null };
for (const t of times) {
  await g.page.evaluate((t) => window.__game.seek(t), t);
  const cam = await g.page.evaluate(() => window.__game.cameraInfo());
  const frame = { t, files: {}, camera: cam };
  for (const pass of passes) {
    const b64 = await g.page.evaluate((p) => window.__game.render(p), pass);
    const file = `${pass}_${t.toFixed(3)}.png`; savePng(b64, path.join(outDir, file)); frame.files[pass] = file;
  }
  frame.state = await g.page.evaluate(() => window.__game.state());
  index.frames.push(frame);
}
index.camera_far = await g.page.evaluate(() => window.__game.program?.camera?.intrinsics?.far ?? 100);
index.browser_events = [...g.logs];
index.elapsed_ms = Date.now() - t0;
fs.writeFileSync(path.join(outDir, 'index.json'), JSON.stringify(index, null, 2));
await g.close();
console.log(`rendered ${times.length} frames x ${passes.length} passes to ${outDir} in ${index.elapsed_ms} ms`);
