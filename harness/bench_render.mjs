// bench_render.mjs — 两轴评测：按给定的一组相机渲染候选场景。
//
// 用法:
//   node harness/bench_render.mjs --cand DIR --cameras FILE --out DIR
//        [--mode flat|textured] [--width W --height H] [--names a,b,c] [--limit N]
//
// --cand     候选目录：scene.js、materials.json（可选）、textures/（textured 模式用）
// --cameras  { views: [{ name, pos, quat, fov_deg, aspect, width?, height? }] }
// 画布尺寸默认取第一台相机的 width / height，这样留出视角的渲染图和照片逐像素对得上。
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { chromiumArgs, parseArgs, savePng } from './common.mjs';

const REPO = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript',
               '.json': 'application/json', '.png': 'image/png', '.jpg': 'image/jpeg' };

/** 三个前缀各指向一个目录，别的路径一律 404。候选目录是不可信输入，不让它够到外面。 */
export function serveBench(candDir) {
  const routes = { '/kernel/': path.join(REPO, 'kernel'), '/three/': path.join(REPO, 'node_modules', 'three'),
                   '/cand/': path.resolve(candDir) };
  const server = http.createServer((req, res) => {
    const url = decodeURIComponent(req.url.split('?')[0]);
    const prefix = Object.keys(routes).find(p => url.startsWith(p));
    const root = prefix ? routes[prefix] : null;
    const file = root ? path.join(root, url.slice(prefix.length)) : null;
    if (!file || !(file === root || file.startsWith(root + path.sep)) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
      res.writeHead(404); res.end('not found'); return;
    }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    fs.createReadStream(file).pipe(res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve({ server, url: `http://127.0.0.1:${server.address().port}` })));
}

export async function renderViews({ cand, views, out, mode = 'flat', width, height, timeoutMs = 120000 }) {
  fs.mkdirSync(out, { recursive: true });
  const started = Date.now();
  const { server, url } = await serveBench(cand);
  const browser = await chromium.launch({ headless: true, args: chromiumArgs(), timeout: 90000 });
  const logs = [];
  try {
    const page = await browser.newPage({ viewport: { width, height } });
    page.on('console', m => logs.push(`[${m.type()}] ${m.text()}`));
    page.on('pageerror', e => logs.push(`[pageerror] ${e.message}`));
    await page.goto(`${url}/kernel/bench_view.html?mode=${mode}&w=${width}&h=${height}`, { waitUntil: 'load', timeout: 60000 });
    await page.waitForFunction(() => window.__bench && window.__bench.readyState !== 'loading', null, { timeout: timeoutMs });
    const state = await page.evaluate(() => ({ readyState: window.__bench.readyState, error: window.__bench.error,
                                               objects: window.__bench.objects, materials_used: window.__bench.materials_used }));
    if (state.readyState !== 'ready') throw new Error(`场景加载失败：${state.error}\n${logs.join('\n')}`);
    const frames = [];
    for (const view of views) {
      const b64 = await page.evaluate(v => window.__bench.render(v), view);
      const file = `${path.parse(view.name).name}.png`;
      savePng(b64, path.join(out, file));
      frames.push({ name: view.name, file });
    }
    const index = { mode, width, height, frames, objects: state.objects, materials_used: state.materials_used,
                    browser_events: logs, elapsed_ms: Date.now() - started };
    fs.writeFileSync(path.join(out, 'index.json'), JSON.stringify(index, null, 1));
    return index;
  } finally {
    await browser.close();
    server.close();
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const args = parseArgs(process.argv.slice(2));
  for (const key of ['cand', 'cameras', 'out']) if (!args[key]) throw new Error(`缺少 --${key}`);
  let views = JSON.parse(fs.readFileSync(args.cameras, 'utf8')).views;
  if (args.names) { const want = new Set(args.names.split(',')); views = views.filter(v => want.has(v.name) || want.has(path.parse(v.name).name)); }
  if (args.limit) views = views.slice(0, Number(args.limit));
  if (!views.length) throw new Error('没有要渲染的视角');
  const width = Number(args.width ?? views[0].width ?? 640);
  const height = Number(args.height ?? views[0].height ?? Math.round(width / (views[0].aspect ?? 1.5)));
  const index = await renderViews({ cand: args.cand, views, out: args.out, mode: args.mode ?? 'flat', width, height });
  console.log(`rendered ${index.frames.length} views (${index.mode}, ${width}x${height}) to ${args.out} in ${index.elapsed_ms} ms`);
}
