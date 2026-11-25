// bench_sequence.mjs — 两轴评测的连续交互：同一个世界里按顺序做一串动作，中途不重置。
//
// bench_interact.mjs 的清单每条各开一个新世界，量的是「这一个物体判得对不对、执行出来成不成立」。
// 这里量的是动作之间的后果：推边几的时候上面的碗跟不跟着走，搬走的杯子后面还在不在原处，
// 没碰过的东西有没有被带乱。这份清单同样不给智能体看。
//
//   node harness/bench_sequence.mjs --cand DIR --sequence FILE --out FILE [--settle] [--anchors FILE]
//
// 物体按标签指，给了 --anchors 时按位置对到场景里的物体上，规则和 bench_interact.mjs 的 resolveTargets 一样。
//
// 默认就是内核原本的样子：没被碰到的物体是焊死的，只有手上那个、以及它直接碰到的会被唤醒
// （kernel/manipulate.js 的 wakeContacts）。--settle 则在开始前把所有能动的物体交给物理落稳一遍，
// 之后它们一直由仿真驱动。两种条件对照着跑，才分得清失败是场景写得不对，还是运行时只唤醒直接接触者。
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import * as THREE from 'three';
import { initRapier } from '../kernel/physics.js';
import { buildBenchScene, makeContext } from '../kernel/bench_scene.js';
import { interactionOf } from '../kernel/manipulate.js';
import { makeWorldFrom } from './interaction_probe.mjs';
import { loadCandidate, resolveTargets } from './bench_interact.mjs';
import { parseArgs } from './common.mjs';

const MOVE_FRAMES = 60;      // 每段移动用 1 秒走完，缓起缓停：手是缓缓推的，不是猛拽
const HOLD_FRAMES = 20;
const SETTLE_FRAMES = 150;   // 松手后 2.5 秒
const ON_GAP_M = 0.03;       // 物体底面离正下方那个东西不超过 3 厘米，才算放在它上面
const STAYED_M = 0.02;       // 没碰过的东西挪了不到 2 厘米算没动
const FOLLOW_TOL = 0.3;      // 被推的东西停下的位置，偏差不超过请求距离的 30%
const UPRIGHT_DEG = 15;
const CARRY_HEIGHT = 1.2;    // 端着东西走时抬到这个高度以上，越过桌面、沙发和椅背，不去扫别的东西

const round = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);

function buildWorld({ module, materials }) {
  return makeWorldFrom(buildBenchScene(module, makeContext({ materials, mode: 'flat' })));
}

function entryOf(world, label) {
  const id = world.labels.get(label)?.id;
  const entry = world.registry.find(e => e.id === id);
  if (!entry) throw new Error(`场景里找不到 ${label}`);
  return entry;
}

function labelsOf(sequence) {
  const out = new Set();
  const add = (c) => {
    for (const key of ['followed', 'upright', 'turned']) if (c[key]) out.add(c[key]);
    for (const id of [...(c.on ?? []), ...(c.stayed ?? [])]) out.add(id);
  };
  for (const step of sequence.steps) {
    out.add(step.target);
    if (step.onto) out.add(step.onto);
    step.expect.forEach(add);
  }
  sequence.final.forEach(add);
  return [...out];
}

/** 从物体中心往下打一条射线，第一个碰到的别的物体就是它正下方的支撑。 */
function below(world, entry) {
  world.scene.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(entry.group);
  const origin = box.getCenter(new THREE.Vector3());
  const ray = new THREE.Raycaster(origin, new THREE.Vector3(0, -1, 0));
  const hit = ray.intersectObjects(world.scene.children, true)
    .find(h => h.object.userData.entry && h.object.userData.entry !== entry);
  if (!hit) return { under: null, gap: null };
  return { under: hit.object.userData.entry.id, gap: box.min.y - hit.point.y };
}

/** 支撑物在 (x, z) 处上表面的高度。 */
function topAt(world, support, x, z) {
  world.scene.updateMatrixWorld(true);
  const ray = new THREE.Raycaster(new THREE.Vector3(x, 50, z), new THREE.Vector3(0, -1, 0));
  const hit = ray.intersectObject(support.group, true)[0];
  if (!hit) throw new Error(`${support.id} 在 (${x}, ${z}) 处没有上表面`);
  return hit.point.y;
}

// 内核的抓取是给鼠标拖动设计的：抓住期间关掉重力、不管朝向。平推一张悬臂边几时，台面上东西的
// 重量会给它一个力矩，没有重力压着、也没有人稳住，它就慢慢转起来，把台面上的东西甩出去。
// 人或机械臂推家具、端杯子时是稳住朝向的，所以这里每一帧用角速度把它拉回抓住那一刻的朝向，
// 和内核里柜门对齐目标朝向的做法一样。
function steady(entry, quat) {
  const delta = quat.clone().multiply(entry.group.quaternion.clone().invert());
  if (delta.w < 0) { delta.x = -delta.x; delta.y = -delta.y; delta.z = -delta.z; delta.w = -delta.w; }
  const angle = 2 * Math.acos(Math.min(1, delta.w));
  const sin = Math.sqrt(Math.max(0, 1 - delta.w * delta.w));
  const spin = sin < 1e-9 ? new THREE.Vector3() : new THREE.Vector3(delta.x, delta.y, delta.z).multiplyScalar(angle / sin * 60);   // 一帧（1/60 秒）转回去
  entry.body.setAngvel({ x: spin.x, y: spin.y, z: spin.z }, true);
}

function glide(world, entry, quat, from, to) {
  for (let i = 1; i <= MOVE_FRAMES; i++) {
    const t = i / MOVE_FRAMES;
    const p = from.clone().lerp(to, t * t * (3 - 2 * t));   // 缓起缓停，起步时速度不跳变
    world.manipulator.dragToPoint(p.x, p.y, p.z);
    steady(entry, quat);
    world.step(1);
  }
}

function perform(world, step) {
  const entry = entryOf(world, step.target);
  const start = entry.group.position.clone();
  const quat = entry.group.quaternion.clone();
  if (!world.manipulator.grab(entry, start.clone())) return { grabbed: false };
  if (step.action === 'push') {
    glide(world, entry, quat, start, start.clone().add(new THREE.Vector3(step.by[0], 0, step.by[1])));
  } else if (step.action === 'carry') {
    const lift = start.y - new THREE.Box3().setFromObject(entry.group).min.y;   // 原点离底面多高
    const top = topAt(world, entryOf(world, step.onto), step.at[0], step.at[1]);
    const high = Math.max(start.y + 0.2, top + lift + 0.2, CARRY_HEIGHT);
    const up = start.clone().setY(high);
    const over = new THREE.Vector3(step.at[0], high, step.at[1]);
    glide(world, entry, quat, start, up);
    glide(world, entry, quat, up, over);
    glide(world, entry, quat, over, new THREE.Vector3(step.at[0], top + lift + 0.01, step.at[1]));
  } else if (step.action === 'turn') {
    const { axis, pivot } = interactionOf(entry);
    const toRest = start.clone().sub(pivot);
    const flat = toRest.clone().addScaledVector(axis, -toRest.dot(axis));
    const radius = flat.length();
    const tangent = new THREE.Vector3().crossVectors(axis, flat).normalize();
    flat.normalize();
    for (let i = 1; i <= MOVE_FRAMES; i++) {
      const a = step.angle * i / MOVE_FRAMES;
      const p = pivot.clone().addScaledVector(flat, radius * Math.cos(a))
        .addScaledVector(tangent, radius * Math.sin(a)).addScaledVector(axis, toRest.dot(axis));
      world.manipulator.dragToPoint(p.x, p.y, p.z);
      world.step(1);
    }
  } else {
    throw new Error(`不认识的动作 ${step.action}`);
  }
  for (let i = 0; i < HOLD_FRAMES; i++) {
    if (step.action !== 'turn') steady(entry, quat);
    world.step(1);
  }
  world.manipulator.release();
  world.step(SETTLE_FRAMES);
  return { grabbed: true };
}

function judge(world, check, ctx) {
  if (check.on) {
    const [id, support] = check.on;
    const { under, gap } = below(world, entryOf(world, id));
    const pass = under === support && gap != null && Math.abs(gap) <= ON_GAP_M;
    return { kind: 'on', target: id, expect: support, pass, detail: { under, gap_m: round(gap) } };
  }
  if (check.followed) {
    const moved = entryOf(world, check.followed).group.position.clone().sub(ctx.before.get(check.followed));
    const want = new THREE.Vector3(ctx.step.by[0], 0, ctx.step.by[1]);
    const error = moved.clone().setY(0).sub(want).length() / want.length();
    return { kind: 'followed', target: check.followed, pass: error <= FOLLOW_TOL,
             detail: { moved_m: moved.toArray().map(round), error: round(error) } };
  }
  if (check.upright) {
    const entry = entryOf(world, check.upright);
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(entry.group.quaternion);
    const tilt = THREE.MathUtils.radToDeg(up.angleTo(new THREE.Vector3(0, 1, 0)));
    return { kind: 'upright', target: check.upright, pass: tilt <= UPRIGHT_DEG, detail: { tilt_deg: round(tilt) } };
  }
  if (check.turned) {
    const entry = entryOf(world, check.turned);
    const turned = THREE.MathUtils.radToDeg(entry.group.quaternion.angleTo(ctx.beforeQuat.get(check.turned)));
    return { kind: 'turned', target: check.turned, pass: turned >= check.min_deg, detail: { turned_deg: round(turned) } };
  }
  if (check.stayed) {
    return check.stayed.map(id => {
      const shift = entryOf(world, id).group.position.distanceTo(ctx.initial.get(id));
      return { kind: 'stayed', target: id, pass: shift <= STAYED_M, detail: { shift_m: round(shift) } };
    });
  }
  throw new Error(`不认识的检查 ${JSON.stringify(check)}`);
}

export function runSequence(candidate, sequence, { settle = false, anchors = null } = {}) {
  const world = buildWorld(candidate);
  world.labels = resolveTargets(world.registry, labelsOf(sequence), anchors);
  if (settle) {
    for (const e of world.registry.filter(e => e.kind === 'object')) {
      world.physics.makeDynamic(e);
      e.dynamic = true; e.simulated = true;
    }
    world.step(SETTLE_FRAMES);
  }
  const snapshot = () => new Map([...world.labels].filter(([, r]) => r.id).map(([label]) => [label, entryOf(world, label).group.position.clone()]));
  const initial = new Map([...world.labels].filter(([, r]) => r.id).map(([label, r]) => [label, entryOf(world, label).group.position.clone()]));
  const steps = [];
  for (const step of sequence.steps) {
    const before = snapshot();
    const beforeQuat = new Map([...world.labels].filter(([, r]) => r.id).map(([label]) => [label, entryOf(world, label).group.quaternion.clone()]));
    const wokenBefore = world.manipulator.woken.length;
    const done = perform(world, step);
    const checks = step.expect.flatMap(c => judge(world, c, { step, before, beforeQuat, initial }));
    const moved = entryOf(world, step.target).group;
    const tilt = THREE.MathUtils.radToDeg(new THREE.Vector3(0, 1, 0).applyQuaternion(moved.quaternion).angleTo(new THREE.Vector3(0, 1, 0)));
    steps.push({ action: step.action, target: step.target, ...done,
                 target_after: { pos: moved.position.toArray().map(round), tilt_deg: round(tilt) },
                 woken: world.manipulator.woken.slice(wokenBefore).map(w => w.id), checks });
  }
  const final = sequence.final.flatMap(c => judge(world, c, { initial }));
  const all = [...steps.flatMap(s => s.checks), ...final];
  const passed = all.filter(c => c.pass).length;
  return {
    scene: sequence.scene ?? null,
    condition: settle ? 'settled' : 'lazy',
    resolved: Object.fromEntries([...world.labels].map(([label, r]) => [label, r.id])),
    summary: { passed, total: all.length, pass_rate: round(passed / all.length) },
    steps, final,
  };
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv.slice(2));
  for (const key of ['cand', 'sequence', 'out']) if (!args[key]) throw new Error(`缺少 --${key}`);
  await initRapier();
  const candidate = await loadCandidate(args.cand);
  const sequence = JSON.parse(fs.readFileSync(args.sequence, 'utf8'));
  const anchors = args.anchors ? JSON.parse(fs.readFileSync(args.anchors, 'utf8')).anchors : null;
  const report = runSequence(candidate, sequence, { settle: args.settle === 'true', anchors });
  fs.mkdirSync(path.dirname(path.resolve(args.out)), { recursive: true });
  fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
  const s = report.summary;
  console.log(`sequence (${report.condition}): ${s.passed}/${s.total} passed (${s.pass_rate})`);
  for (const [i, step] of report.steps.entries()) {
    console.log(`  step ${i + 1} ${step.action} ${step.target}  woken=[${step.woken.join(',')}]`);
    for (const c of step.checks.filter(c => !c.pass)) console.log(`    FAIL ${c.kind} ${c.target} ${JSON.stringify(c.detail)}`);
  }
  for (const c of report.final.filter(c => !c.pass)) console.log(`  FINAL FAIL ${c.kind} ${c.target} ${JSON.stringify(c.detail)}`);
}
