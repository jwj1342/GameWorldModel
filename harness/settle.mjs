// settle.mjs — 把编译好的场景放进物理里跑一会儿，让物体落到支撑面上、解开互相穿插，
// 然后把稳定之后的位姿写出来。
//
// 为什么要这一步：感知给出的位姿是按**可见表面**拟合的包围盒中心，物体常常半悬空，
// 或者和桌面、和别的物体穿插。这种场景看着就不对，而且一旦用鼠标去拖，物体会从
// 别的东西里被拽出来。先让它自己稳一稳，拖起来才像那么回事。
//
// 用法: node_harness.sh settle.mjs --game <目录> --out <json> [--seconds 1.5]
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const seconds = Number(args.seconds ?? 1.5);
const g = await openGame(args.game, { width: 64, height: 64, logFile: path.join(path.dirname(args.out), 'settle.log') });
const report = await g.page.evaluate((s) => window.__game.settle(s), seconds);
fs.mkdirSync(path.dirname(args.out), { recursive: true });
fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
const moved = report.settled.filter(r => r.moved > 0.01);
console.log(`稳定 ${seconds}s：${report.settled.length} 个物体参与，${moved.length} 个移动超过 1 cm，`
  + `最大位移 ${Math.max(0, ...report.settled.map(r => r.moved)).toFixed(3)} m`);
await g.close();
