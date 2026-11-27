// bench_scene.js — 两轴评测里，把智能体写的场景代码变成内核认识的场景和注册表。
//
// 智能体交一个模块：
//   export function describe(THREE, ctx) -> [{ id, class?, kind, object3D, mass_kg?, interaction? }]
//   export function environment(THREE)   -> { background?, lights? }        （可选）
//
// kind 只有两种：'static'（墙、地面、搬不动的东西）和 'movable'（能被拿起来、推开的东西）。
// 这是行为那一半里智能体唯一要做的判断，评测的交互清单查的就是它判得对不对、执行出来成不成立。
//
// 样式和行为在这里就分开：
//   - 外观走 ctx.material(name)，名字对应 materials.json 里的一条「文字描述 + 种子 + 底色」。
//     同一份场景代码可以用纯色渲染，也可以贴上生成的贴图渲染，代码一个字不改。
//   - 碰撞体只看几何，每个网格一个轴对齐的盒子，和材质无关。没有画面的 Node 里也能算，
//     交互探针就是这么跑的。
//
// 和 scene_from_code.js 的关系：约定是同一个思路（模型写 describe，管道补齐碰撞体和编号），
// 区别是这里多了 ctx，以及按网格拆碰撞体。那边是给 DSL 对照实验用的，一个物体只给一个包围盒，
// 桌子底下就塞不进东西了。
import * as THREE from 'three';

export const MATERIAL_NAME = /^[a-z0-9_]+$/;
const MIN_EXTENT = 0.01;

/** 给场景代码用的上下文。mode 为 'flat' 时只用底色；'textured' 时贴 loadTexture 给的贴图。 */
export function makeContext({ materials = {}, mode = 'flat', loadTexture = null } = {}) {
  if (!['flat', 'textured'].includes(mode)) throw new Error(`不认识的材质模式 ${mode}`);
  const used = new Set();
  return {
    mode,
    used,
    material(name, { repeat = [1, 1], ...overrides } = {}) {
      if (typeof name !== 'string' || !MATERIAL_NAME.test(name)) throw new Error(`材质名只能用小写字母、数字和下划线：${name}`);
      const spec = materials[name];
      if (!spec) throw new Error(`materials.json 里没有 ${name}`);
      used.add(name);
      const params = {
        color: new THREE.Color(spec.base_color ?? '#888888'),
        roughness: spec.roughness ?? 0.8,
        metalness: spec.metalness ?? 0.0,
        ...overrides,
      };
      const material = new THREE.MeshStandardMaterial(params);
      if (mode === 'textured' && loadTexture) {
        const texture = loadTexture(name);
        if (texture) {
          texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
          texture.repeat.set(Number(repeat[0]) || 1, Number(repeat[1]) || 1);
          texture.colorSpace = THREE.SRGBColorSpace;
          material.map = texture;
          // 贴图自己带颜色，底色只在纯色模式下用，叠在一起会把贴图压暗
          material.color.set('#ffffff');
        }
      }
      return material;
    },
  };
}

/** 每个网格一个盒子，算在物体自己的局部坐标里。调用时 group 必须在原点、没有旋转。 */
function collidersOf(group) {
  group.updateMatrixWorld(true);
  const out = [];
  const corner = new THREE.Vector3();
  group.traverse(node => {
    if (!node.isMesh || !node.geometry) return;
    if (!node.geometry.boundingBox) node.geometry.computeBoundingBox();
    const local = node.geometry.boundingBox;
    if (!local || local.isEmpty()) return;
    const box = new THREE.Box3();
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? local.max.x : local.min.x, i & 2 ? local.max.y : local.min.y, i & 4 ? local.max.z : local.min.z);
      box.expandByPoint(corner.applyMatrix4(node.matrixWorld));
    }
    const size = box.getSize(new THREE.Vector3()), centre = box.getCenter(new THREE.Vector3());
    if (![...size.toArray(), ...centre.toArray()].every(Number.isFinite)) throw new Error('网格的包围盒必须有限');
    out.push({ shape: 'box', extent: size.toArray().map(v => Math.max(v, MIN_EXTENT)), offset: centre.toArray() });
  });
  return out;
}

export function buildBenchScene(module, ctx) {
  if (typeof module?.describe !== 'function') throw new Error('场景模块没有导出 describe(THREE, ctx)');
  const scene = new THREE.Scene();
  const env = typeof module.environment === 'function' ? (module.environment(THREE) ?? {}) : {};
  scene.background = new THREE.Color(env.background ?? '#dddde3');
  const lights = Array.isArray(env.lights) ? env.lights : null;
  if (lights) {
    for (const light of lights) {
      if (!light?.isLight) throw new Error('environment().lights 里只能放 THREE 的灯');
      scene.add(light);
      if (light.target && !light.target.parent) scene.add(light.target);
    }
  } else {
    scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.0));
    const sun = new THREE.DirectionalLight(0xffffff, 1.6);
    sun.position.set(6, 12, 8);
    scene.add(sun);
  }

  const described = module.describe(THREE, ctx);
  if (!Array.isArray(described) || !described.length) throw new Error('describe(THREE, ctx) 要返回一个非空数组');
  const registry = [];
  const ids = new Set();
  let idIndex = 1;
  for (const d of described) {
    if (!d?.object3D?.isObject3D) throw new Error('每个条目都要提供 THREE.Object3D');
    if (typeof d.id !== 'string' || !d.id.trim() || ids.has(d.id)) throw new Error(`物体 id 必须非空且不重复：${d?.id}`);
    if (!['static', 'movable'].includes(d.kind)) throw new Error(`${d.id} 的 kind 只能是 static 或 movable`);
    if (d.mass_kg != null && !(Number.isFinite(d.mass_kg) && d.mass_kg > 0)) throw new Error(`${d.id} 的 mass_kg 必须是正数`);
    ids.add(d.id);
    const root = d.object3D;
    if (root.parent) throw new Error(`${d.id} 的 object3D 必须是独立的根节点`);
    const pos = root.position.clone(), quat = root.quaternion.clone().normalize();
    if (![...pos.toArray(), ...quat.toArray()].every(Number.isFinite)) throw new Error(`${d.id} 的位姿必须有限`);
    const group = new THREE.Group();
    group.name = d.id;
    root.position.set(0, 0, 0); root.quaternion.identity();
    group.add(root);                       // 根 scale 和子节点的局部变换都保留
    const colliders = collidersOf(group);
    if (!colliders.length) throw new Error(`${d.id} 里没有带几何的网格`);
    group.position.copy(pos); group.quaternion.copy(quat);
    scene.add(group);
    const base = { pos: pos.toArray(), quat: quat.toArray() };
    const interaction = d.kind === 'static' ? { grab: false } : (d.interaction ?? undefined);
    const entry = {
      id: d.id, name: d.id, kind: d.kind === 'static' ? 'static' : 'object',
      class: d.class ?? d.id, group, colliders, base,
      spec: { id: d.id, class: d.class ?? d.id, interaction, motion: d.mass_kg ? { mass: d.mass_kg } : {} },
      pose: () => ({ pos: new THREE.Vector3(...base.pos), quat: new THREE.Quaternion(...base.quat) }),
      dynamic: false, simulated: false, idIndex, instanceIndex: 0, visible: true, events: [],
    };
    group.traverse(o => { if (o.isMesh) { o.userData.entry = entry; o.castShadow = true; o.receiveShadow = true; } });
    registry.push(entry);
    idIndex++;
  }
  scene.updateMatrixWorld(true);
  return { scene, registry };
}
