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
      const url = new URL(`${baseUrl}/${info.file}`, document.baseURI);
      // Consume the original Fetch response to completion before parsing. The
      // FileLoader progress-stream wrapper can report Chromium ERR_ABORTED even
      // after all model bytes arrived; keep transport and parse failures explicit.
      const response = await fetch(url.href);
      if (!response.ok) throw new Error(`asset HTTP ${response.status}`);
      const bytes = await response.arrayBuffer();
      const gltf = await loader.parseAsync(bytes, new URL('.', url).href);
      gltf.scene.traverse(o => { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; } });
      scenes[id] = gltf.scene;
    } catch (error) {
      failures.push({ id, file: info.file, error: String(error?.message ?? error) });
    }
  }));
  if (failures.length) console.warn('[gwm] asset load failed:', failures);
  return { scenes, failures };
}


/** 把实测到的颜色迁移到资产上，保留资产自身的相对色彩变化。
 *
 * 挂了真网格之后，程序里那条 material 本来是不生效的——内核用 GLB 自带的材质。
 * 消融测出来这件事有代价：牛顿摆那条昏暗的金属场景上，配上彩色低多边形道具之后
 * 重渲染相似度反而掉了（0.033 -> 0.017）。问题不在资产这条路线，在库的风格。
 *
 * 做法不是把整个模型刷成一个颜色（那会把美术做好的多色细节抹平），而是算出资产
 * 当前的平均色，再按「目标色 / 平均色」逐通道缩放。这样一辆红蓝相间的火车被迁到
 * 深灰色时，红的部分仍然比蓝的部分亮，只是整体偏向深灰。倍率要夹住，否则遇到
 * 接近黑色的资产会炸。
 *
 * 下限给到 0.05 而不是 0.25：three.js 会把 `#404040` 这样的 sRGB 十六进制转成线性
 * （约 0.051），而库里的资产多半是明亮的游戏道具。要把一个亮资产迁到昏暗场景里，
 * 倍率本来就得很小——而昏暗场景正是这件事最该起作用的地方（牛顿摆那条）。
 */
export function tintToMeasured(object, hexColour, { maxGain = 3.0, minGain = 0.05 } = {}) {
  const target = new THREE.Color(hexColour);
  const sum = new THREE.Color(0, 0, 0);
  let count = 0;
  object.traverse(node => {
    if (!node.isMesh || !node.material) return;
    for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
      if (material.color) { sum.add(material.color); count++; }
    }
  });
  if (!count) return object;
  const mean = sum.multiplyScalar(1 / count);
  const gain = ['r', 'g', 'b'].map(c => {
    const value = mean[c] > 1e-4 ? target[c] / mean[c] : 1.0;
    return Math.min(maxGain, Math.max(minGain, value));
  });
  object.traverse(node => {
    if (!node.isMesh || !node.material) return;
    const materials = Array.isArray(node.material) ? node.material : [node.material];
    const tinted = materials.map(material => {
      // 克隆：GLB 的材质在多个实例之间共享，直接改会把别的物体也改掉
      const copy = material.clone();
      if (copy.color) {
        copy.color.setRGB(Math.min(1, copy.color.r * gain[0]),
                          Math.min(1, copy.color.g * gain[1]),
                          Math.min(1, copy.color.b * gain[2]));
      }
      return copy;
    });
    node.material = Array.isArray(node.material) ? tinted : tinted[0];
  });
  return object;
}
