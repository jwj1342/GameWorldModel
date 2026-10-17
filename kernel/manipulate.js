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
const PICK_LAYERS = ['object'];

/** 这个物体能不能被抓。静态结构（地面、墙）默认不能，程序里可以显式关掉某个物体。 */
export function canGrab(entry) {
  if (!entry || !entry.body) return false;
  if (entry.spec?.interaction?.grab === false) return false;
  return PICK_LAYERS.includes(entry.kind);
}

export class Manipulator {
  constructor({ camera, registry, physics, scene, dt = 1 / 60 }) {
    this.camera = camera; this.registry = registry; this.physics = physics;
    this.scene = scene; this.dt = dt;
    this.raycaster = new THREE.Raycaster();
    this.grabbed = null;            // { entry, offset, distance, restore }
    this.target = new THREE.Vector3();
    this.log = [];                  // 操纵记录，给评测用：谁在什么时候被抓、被放到哪
  }

  /** 屏幕归一化坐标 (-1..1) 打一条射线，返回命中的 entry 和世界坐标交点。 */
  pick(ndcX, ndcY) {
    this.raycaster.setFromCamera(new THREE.Vector2(ndcX, ndcY), this.camera);
    const hits = this.raycaster.intersectObjects(this.scene.children, true);
    for (const hit of hits) {
      const entry = hit.object?.userData?.entry;
      if (entry && canGrab(entry)) return { entry, point: hit.point.clone() };
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
    this.grabbed = { entry, offset, distance: this.camera.position.distanceTo(point), restore };
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

  /** 每帧在 physics.step() 之前调一次。 */
  update() {
    if (!this.grabbed) return;
    const { entry } = this.grabbed;
    const delta = this.target.clone().sub(entry.group.position);
    const velocity = delta.divideScalar(this.dt);
    if (velocity.length() > MAX_SPEED) velocity.setLength(MAX_SPEED);
    entry.body.setLinvel({ x: velocity.x, y: velocity.y, z: velocity.z }, true);
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
