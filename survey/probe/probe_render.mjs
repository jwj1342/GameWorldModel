// Headless three.js render probe. Usage: node probe_render.mjs <workdir> <outdir> <gpu:0|1>
import { chromium } from 'playwright';
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';

const [workdir, outdir, gpuFlag] = process.argv.slice(2);
const hasGpu = gpuFlag === '1';
fs.mkdirSync(outdir, { recursive: true });

const html = `<!doctype html><html><body style="margin:0">
<script type="importmap">{"imports":{"three":"/node_modules/three/build/three.module.js"}}</script>
<script type="module">
import * as THREE from 'three';
const W=320,H=240;
const canvas=document.createElement('canvas');canvas.width=W;canvas.height=H;document.body.appendChild(canvas);
const gl=canvas.getContext('webgl2');
const info={webgl2:!!gl};
if(gl){const dbg=gl.getExtension('WEBGL_debug_renderer_info');info.renderer=dbg?gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);info.vendor=dbg?gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL):gl.getParameter(gl.VENDOR);}
try{
  const renderer=new THREE.WebGLRenderer({canvas,antialias:false,preserveDrawingBuffer:true});
  renderer.setSize(W,H,false);
  const scene=new THREE.Scene();scene.background=new THREE.Color(0x202030);
  const cam=new THREE.PerspectiveCamera(50,W/H,0.1,100);cam.position.set(2,2,3);cam.lookAt(0,0,0);
  const cube=new THREE.Mesh(new THREE.BoxGeometry(1,1,1),new THREE.MeshStandardMaterial({color:0xff6633}));scene.add(cube);
  scene.add(new THREE.HemisphereLight(0xffffff,0x404040,1.2));const d=new THREE.DirectionalLight(0xffffff,1);d.position.set(3,5,2);scene.add(d);
  // depth pass via override material
  const t0=performance.now();
  for(let i=0;i<30;i++){cube.rotation.y=i*0.1;renderer.render(scene,cam);}
  gl.finish&&gl.finish();
  info.ms_per_frame=(performance.now()-t0)/30;
  const px=new Uint8Array(4);gl.readPixels(W/2,H/2,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);info.center_pixel=Array.from(px);
  // id-mask pass
  scene.overrideMaterial=new THREE.MeshBasicMaterial({color:0x00ff00});renderer.render(scene,cam);
  gl.readPixels(W/2,H/2,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);info.mask_center_pixel=Array.from(px);scene.overrideMaterial=null;
  // depth pass
  scene.overrideMaterial=new THREE.MeshDepthMaterial({depthPacking:THREE.RGBADepthPacking});renderer.render(scene,cam);
  gl.readPixels(W/2,H/2,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);info.depth_center_pixel=Array.from(px);scene.overrideMaterial=null;
  renderer.render(scene,cam);
  info.three_ok=true;info.three_rev=THREE.REVISION;
}catch(e){info.three_ok=false;info.error=String(e&&e.stack||e);}
window.__probe=info;
</script></body></html>`;
fs.writeFileSync(path.join(workdir, 'index.html'), html);

const server = http.createServer((req, res) => {
  const p = path.join(workdir, decodeURIComponent(req.url.split('?')[0] === '/' ? '/index.html' : req.url.split('?')[0]));
  if (!p.startsWith(workdir) || !fs.existsSync(p)) { res.writeHead(404); res.end(); return; }
  res.writeHead(200, { 'content-type': p.endsWith('.js') ? 'text/javascript' : 'text/html' });
  fs.createReadStream(p).pipe(res);
});
await new Promise((r) => server.listen(0, '127.0.0.1', r));
const port = server.address().port;

const configs = {
  swiftshader: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
};
if (hasGpu) {
  configs.angle_vulkan = ['--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--ignore-gpu-blocklist', '--enable-gpu-rasterization'];
  configs.egl = ['--use-gl=egl', '--ignore-gpu-blocklist', '--enable-gpu-rasterization'];
  configs.angle_gl_egl = ['--use-gl=angle', '--use-angle=gl-egl', '--ignore-gpu-blocklist', '--enable-gpu-rasterization'];
}
const base = ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--headless=new', '--disable-gpu-sandbox'];
const results = {};
for (const [name, args] of Object.entries(configs)) {
  const t0 = Date.now();
  try {
    const browser = await chromium.launch({ headless: true, args: [...base, ...args], timeout: 60000 });
    const page = await browser.newPage({ viewport: { width: 320, height: 240 } });
    const logs = [];
    page.on('console', (m) => logs.push(m.text()));
    page.on('pageerror', (e) => logs.push('PAGEERROR ' + e.message));
    await page.goto(`http://127.0.0.1:${port}/`, { waitUntil: 'load', timeout: 30000 });
    await page.waitForFunction(() => window.__probe !== undefined, null, { timeout: 30000 });
    const info = await page.evaluate(() => window.__probe);
    const shot = path.join(outdir, `probe_${name}.png`);
    await page.screenshot({ path: shot });
    results[name] = { ok: true, ...info, screenshot_bytes: fs.statSync(shot).size, logs: logs.slice(0, 10), total_ms: Date.now() - t0 };
    await browser.close();
  } catch (e) {
    results[name] = { ok: false, error: String(e && e.message || e).slice(0, 800), total_ms: Date.now() - t0 };
  }
}
server.close();
console.log('PROBE_RESULTS_JSON ' + JSON.stringify(results, null, 2));
