// interaction_probe.mjs — 真的去抓、去拖、去松手，量出来「可交互」到底成立多少。
//
// 为什么要这个：gwm/feedback/completeness.py 里的可交互程度原来只数程序**声明**了
// 什么——`interaction.grab` 不是 false 就算一个。声明和能用是两回事：物体可能和桌面
// 穿插着生成、碰撞体退化成零厚度、或者质量被尺度算错了几个数量级，这些情况下声明值
// 照样是 1.0，拖一下才会发现拖不动。审稿人问「这个数是量出来的还是写上去的」时，
// 原来的答案是后者。
//
// 做法是在无头物理里把每个物体真走一遍：抓住 -> 提起来 -> 横着拖 -> 松手 -> 等它稳。
// 不开浏览器、不要 WebGL、不加载 GLB，所以很便宜，每次生成都能顺手跑一遍。
//
// 不加载 GLB 这件事本身就是论文第四条主张的一个检查点：碰撞体永远从基本体那一套算
// （见 kernel/scene.js 的 buildNode），所以行为这一半不需要样式那一半就能算出来。
// --style-swap 把这件事量成一个数：换一套完全不同的外形网格，同一串操作的轨迹应该
// 一丝不差。
//
// 用法:
//   node_harness.sh interaction_probe.mjs --game <目录> --out <json> [--style-swap]
import * as THREE from 'three';
import { Physics, initRapier } from '../kernel/physics.js';
import { buildScene } from '../kernel/scene.js';
import { Manipulator, canGrab, interactionOf } from '../kernel/manipulate.js';

const DT = 1 / 60;
const HOLD = 45;        // 每个拖动阶段推多少帧（0.75 秒）
const SETTLE = 150;     // 松手之后给多少帧落稳（2.5 秒）

// 拖多远。按物体自己的尺寸定，不用固定值：对 10 cm 的小火车，25 cm 是把它扔出去；
// 对一张床，25 cm 又几乎看不出动没动。
//
// 尺寸从**碰撞体**上取，不从画面上的包围盒取。碰撞体是行为那一半，换一套外形不会
// 变；取 group 的包围盒的话，探针自己就把样式混进了行为——第一版这么写，style-swap
// 对照立刻报出 1.6 cm 的偏离，而那一厘米完全是这个函数造成的，内核是干净的。
export function dragDistance(entry) {
  let longest = 0;
  for (const c of entry.colliders ?? []) {
    const e = c.extent ?? c.size;
    if (Array.isArray(e)) longest = Math.max(longest, ...e.map(v => Math.abs(Number(v) || 0)));
    if (c.radius) longest = Math.max(longest, Math.abs(c.radius) * 2);
    if (c.height) longest = Math.max(longest, Math.abs(c.height));
  }
  if (!(longest > 0)) longest = 0.2;
  return Math.min(Math.max(longest * 0.6, 0.03), 0.5);
}

function finite(v) { return Number.isFinite(v.x) && Number.isFinite(v.y) && Number.isFinite(v.z); }

/** 搭一个只有物理、没有画面的世界。assetScenes 传进去是为了做样式替换对照。 */
export function makeWorld(program, { assetScenes = null, dt = DT } = {}) {
  const { scene, registry } = buildScene(program, null, assetScenes);
  const physics = new Physics(dt);
  registry.forEach(e => physics.addEntry(e));
  // 玩家胶囊放到很远的地方：kernel 的 step 需要它存在，但它不该碰到场景里的任何东西
  physics.createPlayer(new THREE.Vector3(1000, 1000, 1000));
  const intr = program.camera?.intrinsics ?? {};
  const camera = new THREE.PerspectiveCamera(intr.fov_deg ?? 50, intr.aspect ?? 16 / 9, 0.05, intr.far ?? 100);
  const key = (program.camera?.keyframes ?? [])[0];
  if (key) {
    camera.position.set(...key.pos);
    if (key.quat) camera.quaternion.set(...key.quat).normalize();
  } else {
    camera.position.set(0, 1.5, 3); camera.lookAt(0, 0, 0);
  }
  camera.updateMatrixWorld(true);
  const manipulator = new Manipulator({ camera, registry, physics, scene, dt });
  const bounds = new THREE.Box3().setFromObject(scene);
  const step = (n, trace = null) => {
    for (let i = 0; i < n; i++) {
      manipulator.update();
      physics.step();
      for (const e of registry) physics.syncSimulated(e);
      if (trace) trace.push(registry.map(e => e.group.position.toArray()));
    }
  };
  return { scene, registry, physics, camera, manipulator, step, bounds };
}

/** 从程序里那台相机看过去，这个物体有多少地方能被鼠标点到。
 *
 *  「能拖」首先得「点得到」。被别的东西挡住、或者根本不在画面里的物体，程序里写着
 *  能抓也没用——这一项和下面的拖动分开报。
 *
 *  在物体投影出来的方框里打一小片网格，而不是只打中心一条射线：中心被前面的东西挡住，
 *  边上露出来的部分照样点得到，只看中心会把这种情况误判成完全够不着。 */
const REACH_GRID = 5;

function reachability(world, entry) {
  const box = new THREE.Box3().setFromObject(entry.group);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, visibleDepth = false;
  for (let i = 0; i < 8; i++) {
    const corner = new THREE.Vector3(i & 1 ? box.max.x : box.min.x,
                                     i & 2 ? box.max.y : box.min.y,
                                     i & 4 ? box.max.z : box.min.z).project(world.camera);
    if (corner.z > -1 && corner.z < 1) visibleDepth = true;
    minX = Math.min(minX, corner.x); maxX = Math.max(maxX, corner.x);
    minY = Math.min(minY, corner.y); maxY = Math.max(maxY, corner.y);
  }
  const onScreen = visibleDepth && maxX >= -1 && minX <= 1 && maxY >= -1 && minY <= 1;
  if (!onScreen) return { on_screen: false, clickable: false, clickable_fraction: 0 };
  let tried = 0, got = 0;
  const blockers = new Map();
  for (let ix = 0; ix < REACH_GRID; ix++) {
    for (let iy = 0; iy < REACH_GRID; iy++) {
      const x = minX + (maxX - minX) * (ix + 0.5) / REACH_GRID;
      const y = minY + (maxY - minY) * (iy + 0.5) / REACH_GRID;
      if (Math.abs(x) > 1 || Math.abs(y) > 1) continue;
      tried++;
      const hit = world.manipulator.pick(x, y);
      if (hit && hit.entry === entry) got++;
      else if (hit) blockers.set(hit.entry.id, (blockers.get(hit.entry.id) ?? 0) + 1);
    }
  }
  const worst = [...blockers.entries()].sort((a, b) => b[1] - a[1])[0];
  return { on_screen: true, clickable: got > 0,
           clickable_fraction: tried ? round(got / tried) : 0,
           occluded_by: got ? null : (worst?.[0] ?? null) };
}

/** 抓住的那一刻，这个物体和别的东西插得有多深。
 *
 *  拖不动的首要嫌疑就是这个：感知给的位姿是按可见表面拟合的包围盒中心，物体常常
 *  半陷在桌面里、或者和邻居互相咬住。插着的时候一抓，接触求解会先把它往外顶，
 *  于是你往上拖、它往下走。这一项把「拖不动」和「生成时就插着」挂上钩。
 *
 *  只能在抓住之后量：Rapier 不会给两个 fixed 刚体算接触，抓住把它变成动力学刚体
 *  之后，和周围固定物的接触才会出现在窄相里。 */
function contactState(world, entry) {
  const w = world.physics.world;
  let deepest = 0, by = null, touching = 0;
  for (const handle of entry.colliderHandles ?? []) {
    const collider = w.getCollider(handle);
    if (!collider) continue;
    w.contactPairsWith(collider, (other) => {
      const hit = world.physics.colliderToEntry.get(other.handle);
      if (!hit || hit === entry) return;
      w.contactPair(collider, other, (manifold) => {
        for (let i = 0; i < manifold.numContacts(); i++) {
          const depth = -manifold.contactDist(i);
          if (depth > 1e-4) touching++;
          if (depth > deepest) { deepest = depth; by = hit.id; }
        }
      });
    });
  }
  return { penetration: round(deepest), penetrating_with: by, contacts: touching };
}

/** 自由拖动那一档：提起来、横着拖、松手。 */
function probeFree(world, entry, trace) {
  const start = entry.group.position.clone();
  const others = world.registry.filter(e => e !== entry && e.kind === 'object')
    .map(e => ({ id: e.id, p: e.group.position.clone(), e }));
  const reach = dragDistance(entry);

  world.manipulator.grab(entry, start.clone());
  // 先空推一帧：抓住把它变成了动力学刚体，接触这时候才算得出来
  world.step(1, trace);
  const contact = contactState(world, entry);
  const lift = start.clone().add(new THREE.Vector3(0, reach, 0));
  world.manipulator.dragToPoint(lift.x, lift.y, lift.z);
  world.step(HOLD, trace);
  const lifted = entry.group.position.clone();

  const side = lifted.clone().add(new THREE.Vector3(reach, 0, 0));
  world.manipulator.dragToPoint(side.x, side.y, side.z);
  world.step(HOLD, trace);
  const dragged = entry.group.position.clone();

  world.manipulator.release();
  world.step(SETTLE, trace);
  const rest = entry.group.position.clone();
  const velocity = entry.body ? entry.body.linvel() : { x: 0, y: 0, z: 0 };
  const speed = Math.hypot(velocity.x, velocity.y, velocity.z);

  const disturbed = others.map(o => o.e.group.position.distanceTo(o.p));
  return {
    mode: 'free',
    reach: round(reach),
    // 抓住那一刻就和别的东西插着多深（米）。拖不动的首要嫌疑
    penetration: contact.penetration,
    penetrating_with: contact.penetrating_with,
    contacts: contact.contacts,
    // 提得起来吗：沿请求方向走了多少，除以请求的距离
    lift_progress: round((lifted.y - start.y) / reach),
    // 跟得住吗：横拖结束时离目标点还有多远，除以请求的距离
    follow_error: round(dragged.distanceTo(side) / reach),
    // 松手之后稳得住吗
    rest_speed: round(speed),
    fell_through: rest.y < world.bounds.min.y - 0.5,
    finite: finite(rest),
    rest_drop: round(start.y - rest.y),
    // 拖一个物体把别人撞飞了多少。生成时互相穿插的话，这个数会很大
    max_disturbance: round(Math.max(0, ...disturbed)),
    disturbed_count: disturbed.filter(d => d > 0.02).length,
  };
}

/** 抽屉和柜门那一档：往限位外面拖，看它是不是真的卡在限位上、而且没偏离约束。 */
function probeConstrained(world, entry, constraint, trace) {
  const start = entry.group.position.clone();
  const startQuat = entry.group.quaternion.clone();
  const axis = constraint.axis.clone();
  const limit = constraint.limit;
  world.manipulator.grab(entry, start.clone());

  if (constraint.mode === 'slide') {
    const want = limit ? Math.max(Math.abs(limit[0]), Math.abs(limit[1])) * 2 + 0.2 : 0.5;
    const target = start.clone().addScaledVector(axis, want);
    world.manipulator.dragToPoint(target.x, target.y, target.z);
    world.step(HOLD * 2, trace);
    const moved = entry.group.position.clone().sub(start);
    const along = moved.dot(axis);
    const drift = moved.clone().addScaledVector(axis, -along).length();
    const cap = limit ? Math.max(limit[0], limit[1]) : null;
    world.manipulator.release();
    return { mode: 'slide', travel: round(along), off_axis_drift: round(drift),
             limit: cap == null ? null : round(cap),
             // 拖到限位外面去了没有。超过 2 cm 就算限位没拦住
             overshoot: cap == null ? null : round(Math.max(0, along - cap)),
             moves: Math.abs(along) > 0.01 };
  }

  // hinge：绕轴往一边扫。目标点取「从支点出发、垂直于轴」的切线方向上一段
  const pivot = constraint.pivot ?? start;
  const toRest = start.clone().sub(pivot);
  const flat = toRest.clone().addScaledVector(axis, -toRest.dot(axis));
  const radius = Math.max(flat.length(), 1e-3);
  const tangent = new THREE.Vector3().crossVectors(axis, flat).normalize();
  const want = limit ? Math.max(Math.abs(limit[0]), Math.abs(limit[1])) * 1.5 + 0.3 : 1.2;  // 弧度
  const target = pivot.clone()
    .addScaledVector(flat.clone().normalize(), radius * Math.cos(want))
    .addScaledVector(tangent, radius * Math.sin(want))
    .addScaledVector(axis, toRest.dot(axis));
  world.manipulator.dragToPoint(target.x, target.y, target.z);
  world.step(HOLD * 2, trace);
  const delta = entry.group.quaternion.clone().multiply(startQuat.clone().invert());
  const turned = 2 * Math.acos(Math.min(1, Math.abs(delta.w)));
  const cap = limit ? Math.max(Math.abs(limit[0]), Math.abs(limit[1])) : null;
  world.manipulator.release();
  return { mode: 'hinge', turned_deg: round(turned * 180 / Math.PI),
           limit_deg: cap == null ? null : round(cap * 180 / Math.PI),
           overshoot_deg: cap == null ? null : round(Math.max(0, turned - cap) * 180 / Math.PI),
           radius: round(radius), moves: turned > 0.02 };
}

const round = (v) => (Number.isFinite(v) ? Math.round(v * 1000) / 1000 : null);

/** 一个物体算不算「真的能交互」：拖得动、而且松手之后还在。
 *
 *  刻意**不**把「从第一帧那台相机点得到」算进来。观众可以转视角、可以走过去，
 *  被挡住不等于不能交互。点得到与否单独报一个数，它量的是另一件事。 */
export function succeeded(row) {
  const d = row.drag;
  if (!d) return false;
  if (d.mode === 'free') {
    return d.lift_progress >= 0.5 && d.follow_error <= 0.25
      && d.finite && !d.fell_through && d.rest_speed < 0.5;
  }
  if (d.mode === 'slide') return d.moves && d.off_axis_drift <= 0.02 && (d.overshoot ?? 0) <= 0.02;
  return d.moves && (d.overshoot_deg ?? 0) <= 2.0;
}

export function probeProgram(program, { assetScenes = null, trace = null } = {}) {
  const probe = makeWorld(program, { assetScenes });
  const grabbable = probe.registry.filter(e => canGrab(e)).map(e => e.id);
  const rows = [];
  for (const id of grabbable) {
    // 每个物体单开一个世界，免得前一个探针把场景搅乱了算到下一个头上
    const world = makeWorld(program, { assetScenes });
    const entry = world.registry.find(e => e.id === id);
    const reach_info = reachability(world, entry);
    const constraint = interactionOf(entry);
    const drag = constraint.mode === 'free'
      ? probeFree(world, entry, trace)
      : probeConstrained(world, entry, constraint, trace);
    rows.push({ id, class: entry.class, declared: constraint.mode, reach_info, drag });
  }
  const objects = probe.registry.filter(e => e.kind === 'object').length;
  const ok = rows.filter(succeeded);
  const free = rows.filter(r => r.drag?.mode === 'free');
  const summary = {
    objects,
    declared_grabbable: rows.length,
    clickable: rows.filter(r => r.reach_info.clickable).length,
    on_screen: rows.filter(r => r.reach_info.on_screen).length,
    works: ok.length,
    // 这才是「可交互程度」该报的数：声明能抓的物体里，真的抓得动的比例
    manipulation_success_fraction: rows.length ? round(ok.length / rows.length) : null,
    // 对整个场景而言，有多少物体是真能动的
    interactive_fraction: objects ? round(ok.length / objects) : null,
    median_lift_progress: median(free.map(r => r.drag.lift_progress)),
    median_follow_error: median(free.map(r => r.drag.follow_error)),
    median_clickable_fraction: median(rows.map(r => r.reach_info.clickable_fraction)),
    median_disturbance: median(free.map(r => r.drag.max_disturbance)),
    fell_through: free.filter(r => r.drag.fell_through).length,
  };
  return { summary, objects: rows };
}

function median(values) {
  const v = values.filter(x => Number.isFinite(x)).sort((a, b) => a - b);
  return v.length ? round(v[(v.length - 1) >> 1]) : null;
}

/** 换一套完全不同的外形网格。用合成的环面结而不是真 GLB，是刻意的：
 *  如果碰撞体有一丝从网格漏出来，换成一个形状、尺寸都南辕北辙的东西，
 *  轨迹立刻就会分叉。真 GLB 反而没这么强的区分力，因为它和基本体的外形是接近的。 */
export function standInScenes(program) {
  const out = {};
  for (const node of program.objects ?? []) {
    const mesh = new THREE.Mesh(new THREE.TorusKnotGeometry(0.5, 0.18, 48, 8),
                                new THREE.MeshStandardMaterial({ color: 0xff00ff }));
    out[node.id] = mesh;
  }
  return out;
}

/** 样式换掉之后，行为这一半有没有跟着动。论文第四条主张的那个数。 */
export function styleSwapDeviation(program) {
  const plain = [], swapped = [];
  probeProgram(program, { trace: plain });
  probeProgram(program, { assetScenes: standInScenes(program), trace: swapped });
  const steps = Math.min(plain.length, swapped.length);
  let worst = 0;
  for (let i = 0; i < steps; i++) {
    const a = plain[i], b = swapped[i];
    for (let k = 0; k < Math.min(a.length, b.length); k++) {
      worst = Math.max(worst, Math.hypot(a[k][0] - b[k][0], a[k][1] - b[k][1], a[k][2] - b[k][2]));
    }
  }
  return { compared_steps: steps, steps_plain: plain.length, steps_swapped: swapped.length,
           max_position_deviation: worst };
}

// ---- 命令行 ----
if (import.meta.url === `file://${process.argv[1]}`) {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const args = {};
  for (let i = 2; i < process.argv.length; i++) {
    const a = process.argv[i];
    if (a.startsWith('--')) args[a.slice(2)] = process.argv[i + 1]?.startsWith('--') === false ? process.argv[++i] : 'true';
  }
  await initRapier();
  const file = args.program ?? path.join(args.game, 'program.json');
  const program = JSON.parse(fs.readFileSync(file, 'utf8'));
  const report = probeProgram(program);
  if (args['style-swap']) report.style_swap = styleSwapDeviation(program);
  if (args.out) {
    fs.mkdirSync(path.dirname(args.out), { recursive: true });
    fs.writeFileSync(args.out, JSON.stringify(report, null, 1));
  }
  const s = report.summary;
  console.log(`物体 ${s.objects}，声明能抓 ${s.declared_grabbable}，点得到 ${s.clickable}，真拖得动 ${s.works}`
    + `（${s.manipulation_success_fraction}）`);
  console.log(`  提起进度中位数 ${s.median_lift_progress}，跟随误差 ${s.median_follow_error}，`
    + `带动别人 ${s.median_disturbance} m，掉穿地面 ${s.fell_through}`);
  if (report.style_swap) {
    console.log(`  换一套外形之后轨迹最大偏离 ${report.style_swap.max_position_deviation} m`
      + `（对比了 ${report.style_swap.compared_steps} 步）`);
  }
}
