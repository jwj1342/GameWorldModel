// drag_check.mjs — 在真实浏览器里用真实鼠标事件拖一个物体，看它动没动。
//
// 为什么要单独做这个：无头物理那条路（harness/interaction_probe.mjs）调的是程序接口，
// 证明的是「给定一个抓取动作，物理能把物体送到目标」。而人用的是鼠标，中间还隔着
// 指针事件绑定、画布坐标换算、射线拾取三层，这三层在无头探针里一层都没走到。
// README 里要写「可以用鼠标把东西拖起来」，证据得来自这条路。
//
// 注意内核只在非 headless 模式下才 attach 指针事件，所以这里开的是交互模式。
//
// 用法: node_harness.sh drag_check.mjs --game <目录> [--out <json>] [--pixels 120] [--shots <目录>]
//      --shots 会存三张截图：按下前、拖住时、松手后。给文档配图用。
//      --mode replay 用视频自己的视角而不是第三人称游玩视角。这也是对「切模式时
//      操纵器的相机要跟着换」的检查——不换的话看到的是视频视角、射线却从玩家那台
//      相机出发，点哪儿都打偏。
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const W = Number(args.w ?? 640), H = Number(args.h ?? 360);
const pixels = Number(args.pixels ?? 120);
const mode = args.mode === 'replay' ? 'replay' : 'play';
const g = await openGame(args.game, { width: W, height: H, interactive: true,
                                      logFile: args.out ? path.join(path.dirname(args.out), 'drag.log') : null });

// 先让场景稳一秒再取坐标。开场那几百毫秒里玩家胶囊在下落、第三人称相机跟着动，
// 这时候算出来的屏幕坐标到按下鼠标的时候已经不对了——第一版就是这么写的，
// 结果鼠标按在了空处，一个事件都没触发。
await g.page.evaluate((m) => window.__game.mode(m), mode);
await g.page.waitForTimeout(1000);

// 挑一个在画面里、而且射线真能打中的物体。靠近画面中心的优先：边上的容易被视锥裁掉。
const target = await g.page.evaluate(([w, h]) => {
  const game = window.__game;
  const cam = game.manipulator.camera;
  game.scene.updateMatrixWorld(); cam.updateMatrixWorld();
  const rows = [];
  for (const e of game.registry) {
    if (e.kind !== 'object') continue;
    const ndc = e.group.position.clone().project(cam);
    if (Math.abs(ndc.x) > 0.8 || Math.abs(ndc.y) > 0.8 || ndc.z <= -1 || ndc.z >= 1) continue;
    const hit = game.manipulator.pick(ndc.x, ndc.y);
    if (!hit || hit.entry !== e) continue;              // 被挡住或打不中就跳过
    rows.push({ id: e.id, ndc: [ndc.x, ndc.y],
                px: [(ndc.x + 1) / 2 * w, (1 - ndc.y) / 2 * h],
                pos: e.group.position.toArray(),
                score: Math.hypot(ndc.x, ndc.y) });
  }
  rows.sort((a, b) => a.score - b.score);
  return rows[0] ?? null;
}, [W, H]);

if (!target) {
  const out = { ok: false, reason: '开场视角里没有一个射线打得中的物体' };
  if (args.out) fs.writeFileSync(args.out, JSON.stringify(out, null, 1));
  console.log(out.reason);
  await g.close();
  process.exit(0);
}

const shot = async (name) => {
  if (!args.shots) return;
  fs.mkdirSync(args.shots, { recursive: true });
  await g.page.screenshot({ path: path.join(args.shots, `${name}.png`) });
};

const [x, y] = target.px;
await shot('1_before');
// 按下之前在同一个像素上复核一次。相机仍然可能在动，这一下能把「鼠标按在空处」
// 和「按对了但拖不动」区分开——两者的现象都是物体没动。
const recheck = await g.page.evaluate(([px, py, w, h]) => {
  const game = window.__game;
  const ndcX = (px / w) * 2 - 1, ndcY = -(py / h) * 2 + 1;
  const hit = game.manipulator.pick(ndcX, ndcY);
  return hit ? hit.entry.id : null;
}, [x, y, W, H]);
await g.page.mouse.move(x, y);
await g.page.mouse.down();
await g.page.mouse.move(x + pixels, y - pixels, { steps: 24 });
await g.page.waitForTimeout(400);                        // 让物理跟上来
await shot('2_held');
const whileHeld = await g.page.evaluate((id) =>
  window.__game.registry.find(e => e.id === id).group.position.toArray(), target.id);
await g.page.mouse.up();
await g.page.waitForTimeout(600);                        // 松手之后落稳
await shot('3_released');
const after = await g.page.evaluate((id) =>
  window.__game.registry.find(e => e.id === id).group.position.toArray(), target.id);
const log = await g.page.evaluate(() => window.__game.grabLog());

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const report = {
  ok: true,
  id: target.id,
  screen_px: [Math.round(x), Math.round(y)],
  drag_px: pixels,
  before: target.pos.map(v => +v.toFixed(4)),
  while_held: whileHeld.map(v => +v.toFixed(4)),
  after_release: after.map(v => +v.toFixed(4)),
  moved_while_held: +dist(target.pos, whileHeld).toFixed(4),
  // 按下那一刻这个像素上是谁。和 id 不一致说明相机或物体在这期间动了，不是拖不动
  under_cursor_at_press: recheck,
  // 抓取记录能证明指针事件确实走到了 Manipulator，而不是物体碰巧被别的东西推动了
  events: log.map(e => e.event),
  grabbed_id: log.find(e => e.event === 'grab')?.id ?? null,
};
report.grab_fired = report.events.includes('grab') && report.events.includes('release');
report.dragged = report.moved_while_held > 0.01 && report.grabbed_id === target.id;
if (args.out) {
  fs.mkdirSync(path.dirname(args.out), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
}
console.log(`[${mode}] 鼠标拖 ${report.id}：按下在 (${report.screen_px})，移动 ${pixels} 像素，`
  + `物体位移 ${report.moved_while_held} m，事件 ${report.events.join('->')}`);
if (report.dragged) console.log('真实鼠标这条路通的');
else if (recheck !== target.id) console.log(`按下那一刻光标底下是 ${recheck}，不是 ${target.id}——取坐标和按下之间画面动了`);
else console.log('按对了但没拖动——指针事件绑定、拾取或物理有一层没走通');
await g.close();
process.exit(report.dragged ? 0 : 1);
