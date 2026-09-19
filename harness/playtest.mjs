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
fs.writeFileSync(path.join(outDir, 'verdict.json'), JSON.stringify(verdict, null, 2));
await g.close();
console.log(JSON.stringify(verdict));
