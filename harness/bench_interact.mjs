// bench_interact.mjs — 两轴评测的交互那一轴：把一份交互清单逐条执行一遍。
//
// 清单是人对着真实场景写的，每条说一件「在现实里成立」的事，例如
//   { "target": "coffee_table", "expect": "movable" }   茶几推得动
//   { "target": "wall_north",   "expect": "fixed" }     墙推不动
//   { "target": "cushion_left", "expect": "rests" }     垫子放着不会自己掉下去或者弹开
//   { "target": "door",         "expect": "hinge" }     门是绕轴转的，转得动、卡得住限位
// 清单不给智能体看。智能体只拿到物体名单（要用哪些 id），哪些能动、哪些不能动要它自己判断。
// 给了 --anchors 的话，清单里的 target 就只是个标签，按位置去找场景里对应的物体（见 resolveTargets），
// 这时连名单也可以不给智能体，物体拆成多细由它自己决定。
//
// 每一条都是执行出来的，不是数声明：movable 走的是 interaction_probe.mjs 那套
// 抓住、提起、横拖、松手、等它稳；rests 是把物体放开让它在重力下待 2.5 秒看挪没挪。
// 没有画面、不开浏览器，所以外观换成什么都不影响这里的结果。
//
// 用法:
//   node harness/bench_interact.mjs --cand DIR --checklist FILE --out FILE [--anchors FILE] [--match id|anchors]
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import * as THREE from 'three';
import { initRapier } from '../kernel/physics.js';
import { canGrab, interactionOf } from '../kernel/manipulate.js';
import { buildBenchScene, makeContext } from '../kernel/bench_scene.js';
import { makeWorldFrom, probeConstrained, probeFree, succeeded } from './interaction_probe.mjs';
import { parseArgs } from './common.mjs';

export const EXPECTATIONS = ['movable', 'fixed', 'rests', 'hinge', 'slide'];
const SETTLE_FRAMES = 150;          // 2.5 秒
const REST_MAX_SHIFT_M = 0.05;      // 放开之后挪动超过 5 厘米就算没放稳
const REST_MAX_TILT_DEG = 10;

const round = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);

export async function loadCandidate(dir) {
  const file = path.join(path.resolve(dir), 'scene.js');
  if (!fs.existsSync(file)) throw new Error(`候选目录里没有 scene.js：${dir}`);
  // 带上修改时间，同一个进程里重跑改过的场景时不会拿到缓存的旧模块
  const module = await import(`${pathToFileURL(file).href}?v=${fs.statSync(file).mtimeMs}`);
  const materialsFile = path.join(path.resolve(dir), 'materials.json');
  const materials = fs.existsSync(materialsFile) ? JSON.parse(fs.readFileSync(materialsFile, 'utf8')) : {};
  return { module, materials };
}

/** 按位置把清单里的标签对到场景里的物体上。
 *
 *  anchors 是 { 标签: [x, y, z] }。包围盒包住这个点的网格里取包得最紧的那个；一个都不包时取
 *  15 厘米以内最近的；再远就算场景里没有。没给 anchors 时标签就是 id。
 *  两个标签落到同一个物体上记为 merged：现实里分开的两样东西被写成了一个，比如碗和边几焊在一起。 */
const ANCHOR_NEAR_M = 0.15;

export function resolveTargets(registry, labels, anchors) {
  const out = new Map();
  if (!anchors) {
    for (const label of labels) out.set(label, { id: registry.some(e => e.id === label) ? label : null, merged: false });
    return out;
  }
  const boxes = registry.flatMap(entry => {
    const list = [];
    entry.group.updateMatrixWorld(true);
    entry.group.traverse(n => { if (n.isMesh) list.push({ entry, box: new THREE.Box3().setFromObject(n) }); });
    return list;
  });
  const volume = (b) => { const s = b.getSize(new THREE.Vector3()); return s.x * s.y * s.z; };
  for (const label of labels) {
    if (!anchors[label]) throw new Error(`anchors 里没有 ${label}`);
    const point = new THREE.Vector3(...anchors[label]);
    const inside = boxes.filter(b => b.box.containsPoint(point))
      .sort((a, b) => volume(a.box) - volume(b.box));
    let hit = inside[0] ?? null;
    if (!hit) {
      const near = boxes.map(b => ({ ...b, d: b.box.distanceToPoint(point) })).sort((a, b) => a.d - b.d)[0];
      hit = near && near.d <= ANCHOR_NEAR_M ? near : null;
    }
    out.set(label, { id: hit ? hit.entry.id : null, merged: false });
  }
  const count = new Map();
  for (const { id } of out.values()) if (id) count.set(id, (count.get(id) ?? 0) + 1);
  for (const r of out.values()) r.merged = r.id != null && count.get(r.id) > 1;
  return out;
}

/** 每条清单各开一个世界，前一条把场景搅乱了不该算到后一条头上。 */
function freshWorld({ module, materials }) {
  return makeWorldFrom(buildBenchScene(module, makeContext({ materials, mode: 'flat' })));
}

function judgeRests(world, entry) {
  const start = entry.group.position.clone(), startQuat = entry.group.quaternion.clone();
  world.physics.makeDynamic(entry);
  entry.dynamic = true; entry.simulated = true;     // 让 step 把仿真结果读回 group
  world.step(SETTLE_FRAMES);
  const end = entry.group.position.clone();
  const shift = end.distanceTo(start);
  const tilt = THREE.MathUtils.radToDeg(entry.group.quaternion.angleTo(startQuat));
  const finite = [end.x, end.y, end.z].every(Number.isFinite);
  const fell = end.y < world.bounds.min.y - 0.5;
  const pass = finite && !fell && shift <= REST_MAX_SHIFT_M && tilt <= REST_MAX_TILT_DEG;
  const reason = pass ? null : !finite ? 'not_finite' : fell ? 'fell_through' : shift > REST_MAX_SHIFT_M ? 'moved_after_release' : 'tipped_over';
  return { pass, reason, detail: { shift_m: round(shift), tilt_deg: round(tilt), dropped_m: round(start.y - end.y) } };
}

// 「能挪动」怎么试。房间里东西摆得密，朝某一个固定方向拖常常正好顶着墙；重家具是推的，
// 不是提起来的。所以按顺序试：先提起再朝四个水平方向拖，再不提起、朝四个方向平推。
// 有一种走得通就算能挪动，记下是哪一种；每次尝试各开一个世界。
const DIRECTIONS = [[1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1]];
export const MOVE_ATTEMPTS = [true, false].flatMap(lift => DIRECTIONS.map(d => ({ lift, direction: d })));

function followed(drag) {
  const lifted = drag.lift_progress == null || drag.lift_progress >= 0.5;
  return lifted && drag.follow_error <= 0.25 && drag.finite && !drag.fell_through && drag.rest_speed < 0.5;
}

function judgeMovable(candidate, target) {
  let best = null;
  for (const [index, attempt] of MOVE_ATTEMPTS.entries()) {
    const world = freshWorld(candidate);
    const entry = world.registry.find(e => e.id === target);
    const drag = probeFree(world, entry, null, { direction: new THREE.Vector3(...attempt.direction), lift: attempt.lift });
    const record = { ...drag, attempt: { ...attempt, index }, attempts_tried: index + 1 };
    if (followed(drag)) return { pass: true, reason: null, detail: record };
    if (!best || drag.follow_error < best.follow_error) best = record;
  }
  return { pass: false, reason: 'drag_failed', detail: { ...best, attempts_tried: MOVE_ATTEMPTS.length } };
}

export function judgeItem(candidate, item, target = { id: item.target, merged: false }) {
  if (!EXPECTATIONS.includes(item.expect)) throw new Error(`清单里有不认识的 expect：${item.expect}`);
  const world = freshWorld(candidate);
  const entry = world.registry.find(e => e.id === target.id);
  if (!entry) return { pass: false, reason: 'missing_object', detail: null };
  // 和别的标签落在同一个物体上时，只有「推不动」那类还能算数；要动的那几类拖的是一整坨，不算
  if (target.merged && item.expect !== 'fixed') return { pass: false, reason: 'merged', detail: null };
  const grabbable = canGrab(entry);

  if (item.expect === 'fixed') {
    return grabbable ? { pass: false, reason: 'declared_movable', detail: null }
                     : { pass: true, reason: null, detail: { body: 'fixed' } };
  }
  if (!grabbable) return { pass: false, reason: 'declared_static', detail: null };
  if (item.expect === 'rests') return judgeRests(world, entry);

  const constraint = interactionOf(entry);
  if (item.expect === 'movable') {
    if (constraint.mode !== 'free') {
      const drag = probeConstrained(world, entry, constraint, null);
      const pass = succeeded({ drag });
      return { pass, reason: pass ? null : 'drag_failed', detail: drag };
    }
    return judgeMovable(candidate, target.id);
  }
  // hinge / slide：声明的约束类型要对，而且执行出来确实沿约束动、卡得住限位
  if (constraint.mode !== item.expect) return { pass: false, reason: `declared_${constraint.mode}`, detail: null };
  const drag = probeConstrained(world, entry, constraint, null);
  const pass = succeeded({ drag });
  return { pass, reason: pass ? null : 'constraint_failed', detail: drag };
}

export function selfChecklist(candidate) {
  const world = freshWorld(candidate);
  const items = [];
  for (const entry of world.registry) {
    if (!canGrab(entry)) continue;
    const mode = interactionOf(entry).mode;
    items.push({ target: entry.id, expect: mode === 'free' ? 'movable' : mode });
    if (mode === 'free') items.push({ target: entry.id, expect: 'rests' });
  }
  return { scene: 'selfcheck', items };
}

/** 锚点单列的那一个数：每个标签的锚点处是不是对的物体。按 id 判的运行（给过物体名单），对的意思是
 *  锚点处的物体就是智能体用这个 id 命名的那个；按锚点判的运行（没给名单），对的意思是锚点找到了物体、
 *  而且没和别的标签落到同一个物体上。锚点取自第一次运行的场景，所以这个数量的是「摆得离那份场景多近」，
 *  只当摆放位置的诊断看，不并进交互分。 */
export function anchorReport(registry, labels, anchors, match) {
  const at = resolveTargets(registry, labels, anchors);
  const ok = (label) => { const r = at.get(label); return match === 'id' ? r.id === label && !r.merged : r.id != null && !r.merged; };
  return { match, found: labels.filter(ok).length, total: labels.length,
           per_label: Object.fromEntries(labels.map(l => [l, { ...at.get(l), ok: ok(l) }])) };
}

/** match 是 'id'（给过物体名单的运行，按 id 找物体）或 'anchors'（没给名单，只能按位置找）。 */
export function runChecklist(candidate, checklist, anchors = null, match = anchors ? 'anchors' : 'id') {
  const labels = [...new Set(checklist.items.map(i => i.target))];
  const registry = freshWorld(candidate).registry;
  const resolved = resolveTargets(registry, labels, match === 'anchors' ? anchors : null);
  const items = checklist.items.map(item => {
    const target = resolved.get(item.target);
    return { ...item, resolved: target.id, ...judgeItem(candidate, item, target) };
  });
  const byExpect = {};
  for (const item of items) {
    const row = byExpect[item.expect] ??= { passed: 0, total: 0 };
    row.total++; if (item.pass) row.passed++;
  }
  const passed = items.filter(i => i.pass).length;
  const probe = freshWorld(candidate);
  return {
    scene: checklist.scene ?? null,
    summary: { passed, total: items.length, pass_rate: items.length ? round(passed / items.length) : null, by_expect: byExpect },
    objects: probe.registry.map(e => ({ id: e.id, kind: e.kind === 'static' ? 'static' : 'movable', colliders: e.colliders.length })),
    match,
    anchors: anchors ? anchorReport(registry, labels, anchors, match) : null,
    items,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv.slice(2));
  const selfcheck = args.selfcheck === 'true';
  for (const key of selfcheck ? ['cand', 'out'] : ['cand', 'checklist', 'out']) if (!args[key]) throw new Error(`缺少 --${key}`);
  await initRapier();
  const candidate = await loadCandidate(args.cand);
  // --selfcheck：不需要清单，把场景里每个声明成 movable 的物体都抓一遍、放一遍。
  // 这是给写场景的智能体自己用的：它能看到自己的场景执行起来成不成立，
  // 但看不到真正的清单里期望哪些物体能动。
  const checklist = selfcheck ? selfChecklist(candidate) : JSON.parse(fs.readFileSync(args.checklist, 'utf8'));
  const anchors = args.anchors ? JSON.parse(fs.readFileSync(args.anchors, 'utf8')).anchors : null;
  const report = runChecklist(candidate, checklist, anchors, args.match ?? (anchors ? 'anchors' : 'id'));
  fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
  const s = report.summary;
  console.log(`interaction checklist (${report.match}): ${s.passed}/${s.total} passed (${s.pass_rate})`
              + (report.anchors ? `; anchors found ${report.anchors.found}/${report.anchors.total}` : ''));
  for (const item of report.items.filter(i => !i.pass)) console.log(`  FAIL ${item.target} expect=${item.expect} reason=${item.reason}`);
}
