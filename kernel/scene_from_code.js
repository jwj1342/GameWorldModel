// 直接出代码这条消融路径的宿主（issue #11）。
//
// 对照的设计：DSL 路径下模型产出 program.json，scene.js 按固定规则把它解释成场景；
// 这条路径下模型直接写 JS。为了让对比只隔离「声明式数据 vs 命令式代码」这一个变量，
// 两边的其余条件完全相同——同样的 three.js 和 Rapier、同样的渲染通道、
// 同样的玩法模板、同样的相机。模型只负责说清楚场景里有什么、怎么动。
//
// 模型要写的模块导出一个函数：
//   export function describe(THREE) -> [{ id, class, kind, object3D, pose? }]
// 其中 pose(t) 返回 { pos: [x,y,z], quat: [x,y,z,w] }；不给就是静止。
//
// idIndex、ID 颜色、碰撞体这些属于管道，不该让模型写——DSL 那边也是编译器生成的，
// 让模型写会把对比变得不公平。所以下面统一补齐。
import * as THREE from 'three';
import { idToColor } from './scene.js';

function colliderFromObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const centre = box.getCenter(new THREE.Vector3()).sub(obj.position);
  return [{ shape: 'box', halfExtents: [size.x / 2 || 0.05, size.y / 2 || 0.05, size.z / 2 || 0.05],
            offset: [centre.x, centre.y, centre.z] }];
}

export async function buildSceneFromCode(program, kernelCfg) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(program?.style?.background ?? '#dddde3');
  const hemi = new THREE.HemisphereLight(0xffffff, 0x444455, 1.0); scene.add(hemi);
  const sun = new THREE.DirectionalLight(0xffffff, 1.6); sun.position.set(6, 12, 8); sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  sun.shadow.camera.left = sun.shadow.camera.bottom = -25;
  sun.shadow.camera.right = sun.shadow.camera.top = 25;
  scene.add(sun);

  const mod = await import('./model_scene.js');
  if (typeof mod.describe !== 'function') throw new Error('模型写的模块没有导出 describe(THREE)');
  const described = mod.describe(THREE);
  if (!Array.isArray(described)) throw new Error('describe(THREE) 要返回一个数组');

  const registry = [];
  let idIndex = 1;
  for (const d of described) {
    if (!d || !d.object3D) { console.warn('[gwm] 跳过一个没有 object3D 的条目', d); continue; }
    const group = new THREE.Group();
    group.add(d.object3D);
    group.name = d.id ?? `obj_${idIndex}`;
    scene.add(group);
    const base = { pos: group.position.toArray(), quat: group.quaternion.toArray() };
    const userPose = typeof d.pose === 'function' ? d.pose : null;
    const entry = {
      id: group.name, name: group.name, kind: d.kind === 'static' ? 'static' : 'object',
      group, colliders: colliderFromObject(group), base, spec: { id: group.name, class: d.class ?? '' },
      class: d.class ?? (d.kind ?? 'object'),
      pose: (t) => {
        if (!userPose) return { pos: new THREE.Vector3(...base.pos), quat: new THREE.Quaternion(...base.quat) };
        const r = userPose(t) ?? {};
        return { pos: new THREE.Vector3(...(r.pos ?? base.pos)),
                 quat: new THREE.Quaternion(...(r.quat ?? base.quat)).normalize() };
      },
      dynamic: !!userPose, simulated: false,
      idIndex, idColor: idToColor(idIndex), instanceIndex: 0,
      visible: true, events: [],
    };
    idIndex++;
    group.traverse(o => { if (o.isMesh) o.userData.entry = entry; });
    registry.push(entry);
  }
  if (!registry.length) throw new Error('describe(THREE) 一个物体都没产出');
  return { scene, registry, lights: { hemi, sun } };
}
