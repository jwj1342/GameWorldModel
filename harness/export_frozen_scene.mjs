// node harness/export_frozen_scene.mjs --game DIR --out NEW_DIR --times 0,1,2
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { openGame, parseArgs, savePng } from './common.mjs';
import { freezeFrame } from './frozen_scene.mjs';

const args = parseArgs(process.argv.slice(2));
const times = (args.times ?? '0,1,2').split(',').map(Number);
const width = Number(args.width ?? 320), height = Number(args.height ?? 240);
if (args['allow-assets'] !== undefined && !['true','false','1','0'].includes(args['allow-assets'])) {
  throw new Error('--allow-assets expects true/false or 1/0');
}
const allowAssets = ['true','1'].includes(args['allow-assets']);
if (!args.game || !args.out || !times.length || times.some(t => !Number.isFinite(t) || t < 0)
    || !Number.isSafeInteger(width) || width < 1 || !Number.isSafeInteger(height) || height < 1) {
  throw new Error('requires --game, --out, finite non-negative times and positive integer dimensions');
}
fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
fs.mkdirSync(args.out); // Deliberately refuses to overwrite an earlier run.
const source = fs.readFileSync(path.join(args.game, 'program.json'));
const sha256 = data => crypto.createHash('sha256').update(data).digest('hex');
const gameFiles = {};
function inventory(dir) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const file = path.join(dir, entry.name);
    if (entry.isDirectory()) inventory(file);
    else gameFiles[path.relative(args.game, file).replaceAll('\\', '/')] = sha256(fs.readFileSync(file));
  }
}
inventory(args.game);
fs.writeFileSync(path.join(args.out, 'program.json'), source);
const g = await openGame(args.game, { width, height, sceneOnly: true,
  logFile: path.join(args.out, 'browser.log') });
try {
  const frames = [];
  for (const [i, t] of times.entries()) {
    const actual = await g.page.evaluate(t => window.__game.seek(t), t);
    if (Math.abs(actual - t) > 1e-9) throw new Error('time mismatch');
    const frame = await g.page.evaluate(freezeFrame, { width, height, allowAssets });
    frame.requested_t = t;
    frame.rgb = `three_${i.toString().padStart(4, '0')}.png`;
    savePng(await g.page.evaluate(() => window.__game.render('rgb')), path.join(args.out, frame.rgb));
    frames.push(frame);
  }
  if (g.logs.some(line => /^\[(error|pageerror|requestfailed)\]/.test(line))) throw new Error('browser errors; inspect browser.log');
  const packet = { format: 'gwm-frozen-scene', schema_version: 1, width, height,
    asset_policy: allowAssets ? 'explicit_ref_loaded_assets' : 'primitives_only',
    semantics: 'shared evaluated state; no independent physics or motion inference',
    program_sha256: sha256(source), game_file_sha256: gameFiles, frames, browser_events: g.logs };
  fs.writeFileSync(path.join(args.out, 'frozen_scene.json'), JSON.stringify(packet));
} finally { await g.close(); }
