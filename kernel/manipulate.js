// manipulate.js — 直接操纵：用鼠标抓住场景里的物体拖动，松手之后物理接管。
//
// 这是「可交互性」里最基本的一档：拍一段卧室，床和被子能拖；拍一张桌子，桌上的杯子
// 能拿起来。它和另外几档是并列的，不是替代关系：
//   观测运动   —— 复现视频里看到的运动，由 motion 字段描述
//   直接操纵   —— 本文件，用户拖动，物理求解
//   关节交互   —— 抽屉沿轴滑、柜门绕铰链转，有限位（尚未实现）
//   玩法交互   —— 角色走到目标、收集、受伤，由 templates/ 实现
//
// 拖动用速度驱动而不是直接改位姿：直接设位置会让物体穿过别的东西，速度驱动则是
// 推过去，碰撞、堆叠、把别的东西撞开都照常发生——这才是「可交互」该有的样子。
import * as THREE from 'three';

const MAX_SPEED = 12;          // m/s，防止鼠标猛甩时把物体弹飞
const MAX_SPIN = 12;           // rad/s，同理
const PICK_LAYERS = ['object'];

/** 这个物体怎么被抓。三种约束，默认自由拖动。
 *
 *   free   —— 没有约束，杯子、枕头、箱子
 *   slide  —— 沿一根轴平移，有行程限位，抽屉
 *   hinge  —— 绕一根轴转动，有角度限位，柜门
 *
 * 约束写在程序的 `interaction` 字段里，和 `motion` 并列但不是一回事：
 * `motion` 说的是「视频里它自己怎么动」，`interaction` 说的是「用户能怎么动它」。
 */
export function interactionOf(entry) {
  const spec = entry?.spec?.interaction;
  if (spec?.grab === false || spec?.grab === 'none') return null;
  const mode = spec?.grab === 'hinge' || spec?.grab === 'slide' ? spec.grab : 'free';
  if (mode === 'free') return { mode: 'free' };
  const axis = new THREE.Vector3(...(spec.axis ?? [0, 1, 0]));
  if (axis.lengthSq() < 1e-9) return { mode: 'free' };
  axis.normalize();
  const limit = Array.isArray(spec.limit) && spec.limit.length === 2
    ? [Number(spec.limit[0]), Number(spec.limit[1])] : null;
  return { mode, axis, limit, pivot: spec.pivot ? new THREE.Vector3(...spec.pivot) : null };
}

/** 这个物体能不能被抓。静态结构（地面、墙）默认不能，程序里可以显式关掉某个物体。 */
export function canGrab(entry) {
  if (!entry || !entry.body) return false;
  if (!interactionOf(entry)) return false;
  return PICK_LAYERS.includes(entry.kind);
}

/** 这个节点以及它所有祖先都可见吗。 */
function isVisible(object) {
  for (let node = object; node; node = node.parent) if (node.visible === false) return false;
  return true;
}

function clamp(value, limit) {
  if (!limit) return value;
  return Math.min(Math.max(value, Math.min(limit[0], limit[1])), Math.max(limit[0], limit[1]));
}

export class Manipulator {
  constructor({ camera, registry, physics, scene, dt = 1 / 60 }) {
    this.camera = camera; this.registry = registry; this.physics = physics;
    this.scene = scene; this.dt = dt;
    this.raycaster = new THREE.Raycaster();
    this.grabbed = null;            // { entry, offset, distance, restore }
    this.target = new THREE.Vector3();
    this.log = [];                  // 操纵记录，给评测用：谁在什么时候被抓、被放到哪
    this.woken = [];                // 被手里的物体推到、因而转成物理驱动的那些
  }

  /** 屏幕归一化坐标 (-1..1) 打一条射线，返回命中的 entry 和世界坐标交点。 */
  pick(ndcX, ndcY) {
    // 先把世界矩阵刷新一遍。射线检测读的是 matrixWorld，而它只在渲染的时候才会被
    // 更新——浏览器里每帧都渲染所以看不出问题，无头跑的时候没人更新，射线就按
    // 「所有东西都还在原点」算，于是点中的是另一个物体。这一下只在点击时发生。
    this.scene.updateMatrixWorld();
    this.camera.updateMatrixWorld();
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);
    // 挡在前面的东西要真的挡住。原来这里是「抓不动就跳过、接着往后找」，于是隔着墙、
    // 隔着桌面也能把后面的杯子抓出来——鼠标能穿墙，不是可交互，是穿帮。
    // 取第一个属于某个节点的命中：它抓得动就抓，抓不动就说明被挡住了。
    //
    // 看不见的东西不算挡住，也不能被抓。three 的射线检测**不**看 visible，而模板会
    // 把收集掉的物品隐藏起来；不滤掉的话，一枚吃掉的金币会一直挡着它后面的东西。
    for (const hit of hits) {
      const entry = hit.object?.userData?.entry;
      if (!entry || !isVisible(hit.object)) continue;
      return canGrab(entry) ? { entry, point: hit.point.clone() } : null;
    }
    return null;
  }

  grab(entry, point) {
    if (!canGrab(entry)) return false;
    this.release();
    const restore = { dynamic: entry.dynamic, simulated: entry.simulated };
    this.physics.grabBody(entry);
    // 抓住期间当作物理驱动：脚本运动让位，位姿由仿真读回来
    entry.dynamic = true; entry.simulated = true; entry.grabbed = true;
    const offset = entry.group.position.clone().sub(point);
    const constraint = interactionOf(entry);
    // 抽屉和柜门要记住静止状态：行程和角度都是相对它量的，松手再抓也不该漂移
    const rest = { pos: entry.group.position.clone(), quat: entry.group.quaternion.clone() };
    this.grabbed = { entry, offset, distance: this.camera.position.distanceTo(point),
                     restore, constraint, rest };
    this.target.copy(entry.group.position);
    this.log.push({ event: 'grab', id: entry.id, at: point.toArray() });
    return true;
  }

  grabAt(ndcX, ndcY) {
    const hit = this.pick(ndcX, ndcY);
    return hit && this.grab(hit.entry, hit.point) ? hit.entry : null;
  }

  /** 把拖动目标设到射线与「过抓取点、垂直于相机朝向」那个平面的交点上。 */
  dragTo(ndcX, ndcY) {
    if (!this.grabbed) return false;
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const ray = this.raycaster.ray;
    const forward = this.camera.getWorldDirection(new THREE.Vector3());
    const anchor = ray.origin.clone().add(forward.clone().multiplyScalar(this.grabbed.distance));
    const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(forward, anchor);
    const point = new THREE.Vector3();
    if (!ray.intersectPlane(plane, point)) return false;
    this.target.copy(point).add(this.grabbed.offset);
    return true;
  }

  /** 直接指定世界坐标目标，给无头测试和脚本化的交互用。 */
  dragToPoint(x, y, z) {
    if (!this.grabbed) return false;
    this.target.set(x, y, z);
    return true;
  }

  /** 把拖动目标投到约束上。自由拖动原样返回；抽屉只保留沿轴的分量并卡住行程；
   *  柜门换算成绕轴的角度并卡住开合范围，同时给出该有的朝向。 */
  constrain(desired) {
    const { constraint, rest } = this.grabbed;
    if (!constraint || constraint.mode === 'free') return { pos: desired, quat: null, value: null };
    const { mode, axis, limit } = constraint;
    if (mode === 'slide') {
      const along = desired.clone().sub(rest.pos).dot(axis);
      const value = clamp(along, limit);
      return { pos: rest.pos.clone().addScaledVector(axis, value), quat: null, value };
    }
    const pivot = constraint.pivot ?? rest.pos;
    // 投到垂直于转轴的平面上，再量从静止位置转过了多少
    const toRest = rest.pos.clone().sub(pivot);
    const toWant = desired.clone().sub(pivot);
    const flat = (v) => v.clone().addScaledVector(axis, -v.dot(axis));
    const a = flat(toRest), b = flat(toWant);
    if (a.lengthSq() < 1e-9 || b.lengthSq() < 1e-9) return { pos: rest.pos.clone(), quat: rest.quat.clone(), value: 0 };
    const sign = Math.sign(a.clone().cross(b).dot(axis)) || 1;
    const raw = sign * a.angleTo(b);
    const value = clamp(raw, limit);
    const turn = new THREE.Quaternion().setFromAxisAngle(axis, value);
    const pos = pivot.clone().add(toRest.clone().applyQuaternion(turn));
    return { pos, quat: rest.quat.clone().premultiply(turn), value };
  }

  /** 手里这个物体推到的东西，也该跟着动起来。
   *
   *  场景里除了手上那个，其余物体默认是**焊死的**：kernel/physics.js 的 addEntry 里，
   *  视频中没观测到运动的物体 dynamic=false，建成 fixed 刚体。于是拿杯子去撞旁边那个
   *  杯子，对方纹丝不动。拖得动一个、推不动别的，不算一个可交互的场景。
   *
   *  这里只激活**真的被碰到**的物体：推到了才活，没碰过的保持原样。所以场景不会一
   *  加载就整体塌下来，开销也只落在接触发生的那一帧。两类物体不碰：
   *  静态结构（地面、墙、桌面）该是焊死的；有脚本运动的物体该按视频里看到的那样走。
   */
  wakeContacts() {
    const { entry } = this.grabbed;
    const world = this.physics?.world;
    if (!entry?.colliderHandles || typeof world?.contactPairsWith !== 'function') return;
    for (const handle of entry.colliderHandles) {
      const collider = world.getCollider(handle);
      if (!collider) continue;
      world.contactPairsWith(collider, (other) => {
        const hit = this.physics.colliderToEntry.get(other.handle);
        if (!hit || hit === entry || hit.kind !== 'object') return;
        if (hit.dynamic || hit.simulated || hit.grabbed) return;
        // contactPairsWith 给的是**可能**接触的一对，真挨上了才算
        let touching = false;
        world.contactPair(collider, other, (manifold) => {
          if (manifold.numContacts() > 0) touching = true;
        });
        if (!touching) return;
        this.physics.makeDynamic(hit);
        hit.dynamic = true; hit.simulated = true;
        this.woken.push({ id: hit.id, by: entry.id });
        this.log.push({ event: 'wake', id: hit.id, by: entry.id });
      });
    }
  }

  /** 每帧在 physics.step() 之前调一次。 */
  update() {
    if (!this.grabbed) return;
    this.wakeContacts();            // 用的是上一步算出来的接触，晚一帧，60Hz 下看不出来
    const { entry } = this.grabbed;
    const goal = this.constrain(this.target);
    this.grabbed.value = goal.value;
    const velocity = goal.pos.clone().sub(entry.group.position).divideScalar(this.dt);
    if (velocity.length() > MAX_SPEED) velocity.setLength(MAX_SPEED);
    entry.body.setLinvel({ x: velocity.x, y: velocity.y, z: velocity.z }, true);
    if (!goal.quat) return;
    // 朝向也用角速度推，不直接改——这样柜门扫过去仍然会撞到挡路的东西
    const delta = goal.quat.clone().multiply(entry.group.quaternion.clone().invert());
    const angle = 2 * Math.acos(Math.min(1, Math.abs(delta.w)));
    const sin = Math.sqrt(Math.max(0, 1 - delta.w * delta.w));
    if (angle < 1e-6 || sin < 1e-9) { entry.body.setAngvel({ x: 0, y: 0, z: 0 }, true); return; }
    const flip = delta.w < 0 ? -1 : 1;
    const spin = new THREE.Vector3(delta.x, delta.y, delta.z)
      .multiplyScalar(flip * angle / sin / this.dt);
    if (spin.length() > MAX_SPIN) spin.setLength(MAX_SPIN);
    entry.body.setAngvel({ x: spin.x, y: spin.y, z: spin.z }, true);
  }

  release() {
    if (!this.grabbed) return null;
    const { entry, restore } = this.grabbed;
    this.physics.releaseBody(entry);
    entry.grabbed = false;
    // 松手之后保持物理驱动：杯子该掉到桌上，而不是停在半空。被拖过的物体从此
    // 由仿真决定位姿，reset() 会把它放回初始位置。
    entry.dynamic = true; entry.simulated = true;
    entry.restoredFrom = restore;
    this.log.push({ event: 'release', id: entry.id, at: entry.group.position.toArray() });
    this.grabbed = null;
    return entry;
  }

  /** 把鼠标事件接到画布上。无头跑的时候不调这个，直接用上面的程序接口。 */
  attach(domElement) {
    const ndc = (event) => {
      const rect = domElement.getBoundingClientRect();
      return [((event.clientX - rect.left) / rect.width) * 2 - 1,
              -((event.clientY - rect.top) / rect.height) * 2 + 1];
    };
    this._handlers = {
      pointerdown: (e) => { if (this.grabAt(...ndc(e))) domElement.setPointerCapture?.(e.pointerId); },
      pointermove: (e) => { if (this.grabbed) this.dragTo(...ndc(e)); },
      pointerup: () => this.release(),
      pointercancel: () => this.release(),
    };
    for (const [name, fn] of Object.entries(this._handlers)) domElement.addEventListener(name, fn);
    this._domElement = domElement;
  }

  detach() {
    if (!this._domElement || !this._handlers) return;
    for (const [name, fn] of Object.entries(this._handlers)) this._domElement.removeEventListener(name, fn);
    this._domElement = null; this._handlers = null;
  }
}
