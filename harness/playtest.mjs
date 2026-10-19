// playtest.mjs — 自动试玩：朝目标行走，受阻则跳，记录 state 与截图，规则判定
// 用法: node playtest.mjs --game DIR --out DIR [--seconds 60] [--shot-every 0.5]
import fs from 'node:fs';
import path from 'node:path';
import { openGame, parseArgs, savePng } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const gameDir = args.game, outDir = args.out;
const seconds = Number(args.seconds ?? 60), shotEvery = Number(args['shot-every'] ?? 0.5);
fs.mkdirSync(path.join(outDir, 'frames'), { recursive: true });
const g = await openGame(gameDir, { width: 640, height: 360, logFile: path.join(outDir, 'browser.log') });
await g.page.evaluate(() => { window.__game.mode('play'); window.__game.reset(0); });
const dt = 1 / 60, steps = Math.round(seconds / dt);
const states = fs.createWriteStream(path.join(outDir, 'states.jsonl'));
let lastShot = -1, stuckSince = 0, lastPos = null, jumpUntil = -1, verdict = { reached_goal: false, stuck: false, fell: 0 };
const wall0 = Date.now();
for (let i = 0; i <= steps; i++) {
  const t = i * dt;
  // autopilot: run in the page for speed (one evaluate per k steps)
  const s = await g.page.evaluate(({ dt, k }) => {
    const gme = window.__game; let st = null;
    for (let j = 0; j < k; j++) {
      st = gme.state(); const p = st.player.pos, goal = st.player.goal.pos;
      const dx = goal[0] - p[0], dz = goal[2] - p[2], d = Math.hypot(dx, dz);
      const move = d > 0.2 ? [dx / d, dz / d] : [0, 0];
      const pilot = window.__pilot || (window.__pilot = { lastProgressT: 0, lastD: d, side: 1 });
      const blockedFor = gme.t - pilot.lastProgressT;
      const jump = blockedFor > 0.6 || (goal[1] - p[1] > 0.8 && goal[1] - p[1] < 1.3 && d < 3.0);
      // when blocked for a while, sidestep perpendicular to the goal direction (alternating side every 2 s)
      if (blockedFor > 1.5) { const side = Math.floor(blockedFor / 2) % 2 === 0 ? pilot.side : -pilot.side; move[0] = move[0] * 0.4 + (-dz / (d || 1)) * side; move[1] = move[1] * 0.4 + (dx / (d || 1)) * side; }
      gme.setInput({ move, jump });
      gme.step(dt);
      if (d < pilot.lastD - 0.02) { pilot.lastProgressT = gme.t; pilot.lastD = d; }
      if (d > pilot.lastD + 1.0) { pilot.lastD = d; pilot.lastProgressT = gme.t; pilot.side = -pilot.side; } // respawned or pushed away
      if (st.player.won) break;
    }
    return gme.state();
  }, { dt, k: 6 });
  i += 5;
  const tt = s.t;
  states.write(JSON.stringify({ t: tt, player: s.player }) + '\n');
  if (tt - lastShot >= shotEvery) { const b64 = await g.page.evaluate(() => window.__game.render('rgb')); savePng(b64, path.join(outDir, 'frames', `rgb_${tt.toFixed(2)}.png`)); lastShot = tt; }
  if (s.player.won) { verdict.reached_goal = true; verdict.t_goal = s.player.won_at; break; }
  const pos = s.player.pos;
  if (lastPos && Math.hypot(pos[0] - lastPos[0], pos[2] - lastPos[2]) < 0.05) { stuckSince += dt * 6; } else stuckSince = 0;
  lastPos = pos;
  if (stuckSince > 20) { verdict.stuck = true; break; }
}
states.end();
const finalState = await g.page.evaluate(() => window.__game.state());
verdict.deaths = finalState.player.deaths; verdict.collected = finalState.player.collected; verdict.collectibles_total = finalState.player.collectibles_total;
verdict.sim_seconds = finalState.t; verdict.wall_ms = Date.now() - wall0; verdict.sim_fps = finalState.frame / Math.max(verdict.wall_ms / 1000, 1e-3);
verdict.playable = verdict.reached_goal && verdict.deaths < 10;
// 这次判定是怎么来的。槽位由模板猜出来的时候，出生点和目标被放在同一块地面的两个
// 角上，中间没有障碍，于是几乎必然通关——那种结果不该和真的走到目标的混在一起算
// 通过率。trustworthy 为 false 的判定，汇总时要单独看。
// 老的 game 目录里打包的是加这个接口之前的内核，取不到就记成 null，不能因此崩掉——
// 判定文件本身比这一项重要，而且归档的产物还要能重跑。
verdict.binding = await g.page.evaluate(
  () => (typeof window.__game.bindingInfo === 'function' ? window.__game.bindingInfo() : null));
// 取不到来源的时候不敢说它可信，但也不能说它不可信——记成 null，由汇总那一步决定怎么办
verdict.trustworthy = verdict.binding
  ? (verdict.binding.inferred_slots ?? []).length === 0
  : null;
if (verdict.trustworthy === false) {
  verdict.untrustworthy_reason =
    `槽位由模板兜底猜出：${verdict.binding.inferred_slots.join('、')}；`
    + `这条链路没走 gwm/binding/，也就没做尺度归一化`;
}
fs.writeFileSync(path.join(outDir, 'verdict.json'), JSON.stringify(verdict, null, 2));
await g.close();
console.log(JSON.stringify(verdict));
