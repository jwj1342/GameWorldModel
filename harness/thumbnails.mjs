// thumbnails.mjs — 给资产库里每个 GLB 渲几张缩略图，供图像检索用。
//
// 为什么要缩略图：光靠文件名做检索会挑错东西（查 "toy train" 会选中 train-connector
// 这种零件而不是机车），因为名字里看不出这个模型长什么样。渲出来之后就可以拿视频里
// 那个物体的裁剪帧直接和它比。做法沿用 docs/design-notes.md 里定下的方案。
//
// 用法:
//   node_harness.sh thumbnails.mjs --library <库根> --out <目录> [--views 4] [--size 160] [--limit N]
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';
import { serveDir, parseArgs, chromiumArgs, savePng } from './common.mjs';

const args = parseArgs(process.argv.slice(2));
const library = path.resolve(args.library);
const outDir = path.resolve(args.out);
const views = Number(args.views ?? 4);
const size = Number(args.size ?? 160);
const limit = args.limit ? Number(args.limit) : Infinity;

const nodeModules = process.env.GWM_NODE_MODULES;
if (!nodeModules) { console.error('需要 GWM_NODE_MODULES 指向 node_modules'); process.exit(2); }

// 临时的伺服根目录：页面 + 指向 three 和资产库的软链。不往库目录里写任何东西。
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'gwm-thumbs-'));
fs.symlinkSync(path.join(nodeModules, 'three'), path.join(root, 'three'));
fs.symlinkSync(library, path.join(root, 'lib'));
fs.writeFileSync(path.join(root, 'index.html'), `<!doctype html><html><head><meta charset="utf-8">
<script type="importmap">{"imports":{"three":"./three/build/three.module.js","three/addons/":"./three/examples/jsm/"}}</script>
</head><body><script type="module">
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setSize(${size}, ${size}, false); renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
document.body.appendChild(renderer.domElement);
const loader = new GLTFLoader();
window.__shoot = async (url, views) => {
  const gltf = await loader.loadAsync(url);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0xf2f2f2);
  scene.add(new THREE.HemisphereLight(0xffffff, 0x888888, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 1.2); sun.position.set(4, 8, 6); scene.add(sun);
  scene.add(gltf.scene);
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 1e-3);
  const camera = new THREE.PerspectiveCamera(40, 1, radius / 100, radius * 100);
  const dist = radius / Math.sin((40 * Math.PI / 180) / 2) * 0.62;
  const shots = [];
  for (let i = 0; i < views; i++) {
    const a = (i / views) * Math.PI * 2 + Math.PI / 4;
    camera.position.set(centre.x + dist * Math.cos(a) * 0.92, centre.y + dist * 0.38, centre.z + dist * Math.sin(a) * 0.92);
    camera.lookAt(centre);
    renderer.render(scene, camera);
    const url = renderer.domElement.toDataURL('image/png');
    shots.push(url.slice(url.indexOf(',') + 1));
  }
  scene.remove(gltf.scene);
  gltf.scene.traverse(o => { if (o.isMesh) { o.geometry?.dispose?.(); } });
  return shots;
};
window.__ready = true;
</script></body></html>`);

const glbs = [];
(function walk(dir) {
  for (const name of fs.readdirSync(dir).sort()) {
    const full = path.join(dir, name);
    const st = fs.statSync(full);
    if (st.isDirectory()) walk(full);
    else if (name.endsWith('.glb')) glbs.push(path.relative(library, full));
  }
})(path.join(library, '3d'));

const wanted = glbs.slice(0, limit);
console.log(`库里 ${glbs.length} 个 GLB，这次渲 ${wanted.length} 个 x ${views} 视角`);

const { server, url } = await serveDir(root);
const browser = await chromium.launch({ headless: true, args: chromiumArgs(), timeout: 90000 });
const page = await browser.newPage({ viewport: { width: size, height: size } });
const failures = [];
page.on('pageerror', e => failures.push({ asset: 'page', error: e.message }));
await page.goto(`${url}/index.html`, { waitUntil: 'load', timeout: 60000 });
await page.waitForFunction(() => window.__ready === true, null, { timeout: 60000 });

fs.mkdirSync(outDir, { recursive: true });
const index = [];
const t0 = Date.now();
for (let i = 0; i < wanted.length; i++) {
  const rel = wanted[i];
  const id = rel.replace(/^3d\//, '').replace(/\.glb$/, '');
  const safe = id.replace(/[^A-Za-z0-9_-]+/g, '-');
  try {
    const shots = await page.evaluate(([u, v]) => window.__shoot(u, v), [`${url}/lib/${rel}`, views]);
    const files = shots.map((b64, k) => { const f = `${safe}_${k}.png`; savePng(b64, path.join(outDir, f)); return f; });
    index.push({ id, path: rel, files });
  } catch (error) {
    failures.push({ asset: id, error: String(error?.message ?? error).slice(0, 200) });
  }
  if ((i + 1) % 250 === 0) {
    const rate = (i + 1) / ((Date.now() - t0) / 1000);
    console.log(`  ${i + 1}/${wanted.length}  ${rate.toFixed(1)} 个/秒  失败 ${failures.length}`);
  }
}
fs.writeFileSync(path.join(outDir, 'thumbs.json'),
  JSON.stringify({ library, views, size, count: index.length, failures, assets: index }, null, 1));
console.log(`完成：${index.length} 个，失败 ${failures.length} 个，用时 ${((Date.now() - t0) / 1000).toFixed(0)} 秒`);
if (failures.length) console.log('前几个失败:', JSON.stringify(failures.slice(0, 5)));
await browser.close(); server.close();
fs.rmSync(root, { recursive: true, force: true });
