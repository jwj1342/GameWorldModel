// 直接出代码这条消融路径的宿主（issue #11）。
//
// 对照的设计：DSL 路径下模型产出 program.json，scene.js 按固定规则把它解释成场景；
// 这条路径下模型直接写 JS。为了让对比只隔离「声明式数据 vs 命令式代码」这一个变量，
// 复用 three.js、Rapier、渲染通道和玩法模板。输入、相机和调用预算仍需单独核对，
// 共享宿主不代表已证明对照条件完全相同。
//
// 模型要写的模块导出一个函数：
//   export function describe(THREE) -> [{ id, class, kind, object3D, pose? }]
// pose(t) 返回根节点的绝对世界位姿，替换初始 position/quaternion，不是再叠加一次。
// 根 scale 与子节点变换属于局部几何；不给 pose 就保留初始根位姿。
//
// idIndex、ID 颜色、碰撞体这些属于管道，不该让模型写——DSL 那边也是编译器生成的，
// 让模型写会把对比变得不公平。所以下面统一补齐。
import * as THREE from 'three';
import { idToColor } from './scene_dsl.js';
// 顶层静态导入，不用 await import。main.js 里 buildScene 是同步调用的，
// 返回 Promise 的话解构出来是 undefined，之前就栽在这。
import { describe } from './model_scene.js';

function colliderFromObject(obj) {
  // 此时根位姿为单位变换，计算的是刚体局部坐标中的包围盒。
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) throw new Error('object3D 没有可用的几何包围盒');
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3());
  if (![...size.toArray(), ...centre.toArray()].every(Number.isFinite)) throw new Error('几何包围盒必须有限');
  return [{ shape: 'box', extent: size.toArray().map(v => v === 0 ? 0.1 : v),
            offset: [centre.x, centre.y, centre.z] }];
}

function checkedPose(pos, quat) {
  if (!Array.isArray(pos) || pos.length !== 3 || !pos.every(Number.isFinite)
      || !Array.isArray(quat) || quat.length !== 4 || !quat.every(Number.isFinite)) {
    throw new Error('pose 必须包含有限的 pos[3] 和 quat[4]');
  }
  const q = new THREE.Quaternion(...quat);
  if (q.lengthSq() === 0 || !Number.isFinite(q.lengthSq())) throw new Error('pose 四元数不能为零或溢出');
  return { pos: new THREE.Vector3(...pos), quat: q.normalize() };
}

export function buildSceneFromCode(program, kernelCfg) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(program?.style?.background ?? '#dddde3');
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444455, 1.0); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(6, 12, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -25;
  sun.shadow.camera.right = sun.shadow.camera.top = 25;
  scene.add(sun);

  if (typeof describe !== 'function') throw new Error('模型写的模块没有导出 describe(THREE)');
  const described = describe(THREE);
  if (!Array.isArray(described)) throw new Error('describe(THREE) 要返回一个数组');

  const registry = [];
  const ids = new Set();
  let idIndex = 1;
  for (const d of described) {
    if (!d?.object3D?.isObject3D) throw new Error('条目必须提供 THREE.Object3D');
    if (d.object3D.parent) throw new Error('object3D 必须是独立根节点，不能带未声明的父变换');
    d.object3D.traverse(node => {
      if (node.matrixAutoUpdate === false || node.matrixWorldAutoUpdate === false) {
        throw new Error('object3D 及子节点必须启用自动矩阵更新；暂不支持手动矩阵控制');
      }
    });
    if (d.kind === 'static' && d.pose != null) throw new Error('static 条目不能提供运动 pose');
    if (d.pose != null && typeof d.pose !== 'function') throw new Error('pose 必须是函数');
    const group = new THREE.Group();
    const root = d.object3D.clone(true);
    const initial = checkedPose(root.position.toArray(), root.quaternion.toArray());
    root.position.set(0, 0, 0); root.quaternion.identity();
    group.add(root); // 保留根 scale 和所有子节点局部变换，不修改 describe 的输入。
    group.name = d.id ?? `obj_${idIndex}`;
    if (typeof group.name !== 'string' || !group.name.trim() || ids.has(group.name)) throw new Error('对象 ID 必须非空且唯一');
    ids.add(group.name);
    const colliders = colliderFromObject(group);
    scene.add(group);
    const base = { pos: initial.pos.toArray(), quat: initial.quat.toArray() };
    const userPose = typeof d.pose === 'function' ? d.pose : null;
    const entry = {
      id: group.name, name: group.name, kind: d.kind === 'static' ? 'static' : 'object',
      group, colliders, base, spec: { id: group.name, class: d.class ?? '' },
      class: d.class ?? (d.kind ?? 'object'),
      pose: (t) => {
        if (!Number.isFinite(t)) throw new Error('pose 时间必须有限');
        if (!userPose) return checkedPose(base.pos, base.quat);
        const r = userPose(t);
        if (!r || typeof r !== 'object') throw new Error('pose(t) 必须返回位姿对象');
        return checkedPose(r.pos ?? base.pos, r.quat ?? base.quat);
      },
      dynamic: !!userPose, simulated: false,
      idIndex, idColor: idToColor(idIndex), instanceIndex: 0,
      visible: true, events: [],
    };
    const start = entry.pose(0);
    group.position.copy(start.pos); group.quaternion.copy(start.quat);
    idIndex++;
    group.traverse(o => { if (o.isMesh) o.userData.entry = entry; });
    registry.push(entry);
  }
  if (!registry.length) throw new Error('describe(THREE) 一个物体都没产出');
  return { scene, registry, lights: { hemi, sun } };
}
