// bench_view.js — 两轴评测的渲染页：加载候选场景，按给定相机出图。
//
// 只做渲染，不做物理，也没有玩法。对外只有 window.__bench：
//   readyState  'loading' | 'ready' | 'error'
//   error       出错时的文字
//   objects     场景里有哪些物体（id、kind）
//   render(cam) 按 { pos, quat, fov_deg, aspect } 渲染一帧，返回 PNG 的 base64
//
// 地址参数：mode=flat|textured|depth，w、h 是画布尺寸。depth 模式不出颜色，出视空间深度：
// 每个像素离相机平面多少毫米，拆成 24 位整数写进 RGB（R 是高 8 位），没有几何的地方是 0。
// 几何误差就靠它和 COLMAP 点比（gwm/bench/geometry.py）。候选场景挂在 /cand/ 下面：
// scene.js、materials.json（可选）、textures/<材质名>.png（textured 模式下必须齐）。
import * as THREE from 'three';
import { buildBenchScene, makeContext } from './bench_scene.js';

const params = new URLSearchParams(location.search);
const bench = { readyState: 'loading', error: null, objects: [] };
window.__bench = bench;

// 视空间深度按毫米编码进 RGB。ShaderMaterial 不经过 three 的色彩空间和色调映射，写进去的就是读出来的
const DEPTH_MATERIAL = new THREE.ShaderMaterial({
  vertexShader: `varying float vDepth;
    void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0); vDepth = -mv.z; gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `varying float vDepth;
    void main() {
      float mm = clamp(floor(vDepth * 1000.0 + 0.5), 0.0, 16777215.0);
      float r = floor(mm / 65536.0), g = floor((mm - r * 65536.0) / 256.0), b = mm - r * 65536.0 - g * 256.0;
      gl_FragColor = vec4(r / 255.0, g / 255.0, b / 255.0, 1.0);
    }`,
});

try {
  const mode = params.get('mode') ?? 'flat';
  const depth = mode === 'depth';
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
  const ctx = makeContext({ materials, mode: depth ? 'flat' : mode, loadTexture });
  const { scene, registry } = buildBenchScene(module, ctx);
  if (requested) await allLoaded;
  if (failed.length) throw new Error(`贴图加载失败：${failed.join(', ')}`);
  // clone() 出来的纹理在原图加载完之后才有图像，这里统一再标一次更新
  scene.traverse(o => { if (o.isMesh && o.material?.map) o.material.map.needsUpdate = true; });

  if (depth) { scene.overrideMaterial = DEPTH_MATERIAL; scene.background = new THREE.Color(0, 0, 0); }
  // 深度图不能抗锯齿：边缘上混出来的颜色解码出来是一个不存在的深度
  const renderer = new THREE.WebGLRenderer({ antialias: !depth, preserveDrawingBuffer: true });
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
