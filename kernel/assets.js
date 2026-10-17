// assets.js — 把编译器解析好的 GLB 读进来，按证据给的尺寸摆正。
//
// 这一层只管「长什么样」。碰撞体仍然由 scene.js 从包围盒生成，和有没有资产无关，
// 所以换网格不会改变任何物体的可拖动性、可踩性或物理行为。场景的表征和交互是两半，
// 这条边界在 gwm/compiler/assets.py 和这里各写了一遍，改的时候两边要一起看。
import * as THREE from 'three';

// 按证据包围盒缩放时，三个轴各缩各的会把模型压扁。我们的包围盒是拟合到**可见表面**的，
// 本身就有系统偏差（见 #7），拿它逐轴拉伸等于把感知误差直接画到画面上。
// 所以默认取三个比例的中位数做等比缩放：尺寸大致对得上，形状不走样。
export function fitToExtent(object, extent, mode = 'uniform') {
  const box = new THREE.Box3().setFromObject(object);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  const want = [extent?.[0], extent?.[1], extent?.[2]].map(v => (Number.isFinite(v) && v > 0 ? v : null));
  const have = [size.x, size.y, size.z];
  const ratios = want.map((w, i) => (w && have[i] > 1e-6 ? w / have[i] : null)).filter(r => r !== null);
  if (!ratios.length) return object;
  let scale;
  if (mode === 'stretch') {
    scale = new THREE.Vector3(...have.map((h, i) => (want[i] && h > 1e-6 ? want[i] / h : median(ratios))));
  } else {
    const s = median(ratios);
    scale = new THREE.Vector3(s, s, s);
  }
  // 先把模型自己的中心挪到原点，再缩放。物体的位姿在程序里指的是包围盒中心，
  // 基本体也是这么摆的，两条路径保持同一个约定。
  object.position.sub(center);
  const holder = new THREE.Group();
  holder.add(object);
  holder.scale.copy(scale);
  return holder;
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const mid = sorted.length >> 1;
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** 读 assets.json 并把里面每个 GLB load 进来。返回 物体id -> THREE.Object3D。
 *  任何一条失败都只是这一条退回基本体，不影响整个场景起不起得来。 */
export async function loadAssets(baseUrl = '.') {
  const manifest = await fetch(`${baseUrl}/assets.json`, { cache: 'no-store' })
    .then(r => (r.ok ? r.json() : null)).catch(() => null);
  const entries = Object.entries(manifest?.objects ?? {});
  if (!entries.length) return { scenes: {}, failures: [] };
  // 动态 import：scene.js 只用这个文件里的 fitToExtent，不该因此被迫解析 GLTFLoader。
  // 编译器扫 three/addons 依赖时认得 import() 这种写法，照样会把它打进 game 目录。
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const loader = new GLTFLoader();
  const scenes = {}, failures = [];
  await Promise.all(entries.map(async ([id, info]) => {
    try {
      const gltf = await loader.loadAsync(`${baseUrl}/${info.file}`);
      gltf.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      scenes[id] = gltf.scene;
    } catch (error) {
      failures.push({ id, file: info.file, error: String(error?.message ?? error) });
    }
  }));
  if (failures.length) console.warn('[gwm] asset load failed:', failures);
  return { scenes, failures };
}
