// bench_view.js — 两轴评测的渲染页：加载候选场景，按给定相机出图。
//
// 只做渲染，不做物理，也没有玩法。对外只有 window.__bench：
//   readyState  'loading' | 'ready' | 'error'
//   error       出错时的文字
//   objects     场景里有哪些物体（id、kind）
//   render(cam) 按 { pos, quat, fov_deg, aspect } 渲染一帧，返回 PNG 的 base64
//
// 地址参数：mode=flat|textured，w、h 是画布尺寸。候选场景挂在 /cand/ 下面：
// scene.js、materials.json（可选）、textures/<材质名>.png（textured 模式下必须齐）。
import * as THREE from 'three';
import { buildBenchScene, makeContext } from './bench_scene.js';

const params = new URLSearchParams(location.search);
const bench = { readyState: 'loading', error: null, objects: [] };
window.__bench = bench;

try {
  const mode = params.get('mode') ?? 'flat';
  const width = Number(params.get('w') ?? 640), height = Number(params.get('h') ?? 360);
  const response = await fetch('/cand/materials.json');
  const materials = response.ok ? await response.json() : {};

  const manager = new THREE.LoadingManager();
  const loader = new THREE.TextureLoader(manager);
  const failed = [];
  let requested = 0;
  const allLoaded = new Promise(resolve => { manager.onLoad = resolve; });
  const cache = new Map();
  const loadTexture = (name) => {
    // 同一张贴图可能被不同的平铺次数用到，图片只下一次，纹理对象各自一份
    if (!cache.has(name)) {
      requested++;
      cache.set(name, loader.load(`/cand/textures/${name}.png`, undefined, undefined, () => failed.push(name)));
      return cache.get(name);
    }
    const copy = cache.get(name).clone();
    copy.needsUpdate = true;
    return copy;
  };

  const module = await import('/cand/scene.js');
  const ctx = makeContext({ materials, mode, loadTexture });
  const { scene, registry } = buildBenchScene(module, ctx);
  if (requested) await allLoaded;
  if (failed.length) throw new Error(`贴图加载失败：${failed.join(', ')}`);
  // clone() 出来的纹理在原图加载完之后才有图像，这里统一再标一次更新
  scene.traverse(o => { if (o.isMesh && o.material?.map) o.material.map.needsUpdate = true; });

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(width, height);
  renderer.shadowMap.enabled = true;
  document.body.appendChild(renderer.domElement);
  const camera = new THREE.PerspectiveCamera(50, width / height, 0.05, 200);

  bench.objects = registry.map(e => ({ id: e.id, kind: e.kind }));
  bench.materials_used = [...ctx.used];
  bench.render = (cam) => {
    camera.fov = cam.fov_deg;
    camera.aspect = cam.aspect ?? width / height;
    camera.position.set(...cam.pos);
    camera.quaternion.set(...cam.quat).normalize();
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    renderer.render(scene, camera);
    return renderer.domElement.toDataURL('image/png').split(',')[1];
  };
  bench.readyState = 'ready';
} catch (error) {
  bench.error = String(error?.stack ?? error);
  bench.readyState = 'error';
}
