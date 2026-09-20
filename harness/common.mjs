// common.mjs — 启动无头 Chromium、静态服务、等待 __game 就绪
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { chromium } from 'playwright';

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.mjs': 'text/javascript', '.json': 'application/json', '.wasm': 'application/wasm', '.png': 'image/png', '.css': 'text/css', '.map': 'application/json' };

export function serveDir(root, port = 0) {
  root = path.resolve(root);
  const server = http.createServer((req, res) => {
    let p = decodeURIComponent(req.url.split('?')[0]); if (p === '/') p = '/index.html';
    const file = path.join(root, p);
    if (!file.startsWith(root) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); res.end('not found ' + p); return; }
    res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream', 'cache-control': 'no-store' });
    fs.createReadStream(file).pipe(res);
  });
  const host = port ? '0.0.0.0' : '127.0.0.1';   // 指定端口通常是给人用的，绑全网卡方便远程访问
  return new Promise((resolve) => server.listen(port, host, () => resolve({ server, url: `http://localhost:${server.address().port}` })));
}

export function chromiumArgs() {
  const base = ['--headless=new', '--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage', '--disable-gpu-sandbox', '--disable-background-timer-throttling', '--disable-renderer-backgrounding', '--ignore-gpu-blocklist'];
  if (process.env.GWM_GPU === '1') return [...base, '--use-gl=angle', '--use-angle=vulkan', '--enable-features=Vulkan,VulkanFromANGLE,DefaultANGLEVulkan', '--enable-gpu-rasterization'];
  return [...base, '--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'];
}

export async function openGame(gameDir, { width = 640, height = 360, logFile = null, timeoutMs = 120000 } = {}) {
  const { server, url } = await serveDir(gameDir);
  const browser = await chromium.launch({ headless: true, args: chromiumArgs(), timeout: 90000 });
  const page = await browser.newPage({ viewport: { width, height } });
  const logs = [];
  const push = (kind, text) => { const line = `[${kind}] ${text}`; logs.push(line); if (logFile) fs.appendFileSync(logFile, line + '\n'); };
  page.on('console', m => push(m.type(), m.text()));
  page.on('pageerror', e => push('pageerror', e.message));
  page.on('requestfailed', r => push('requestfailed', `${r.url()} ${r.failure()?.errorText}`));
  await page.goto(`${url}/index.html?headless=1&w=${width}&h=${height}`, { waitUntil: 'load', timeout: 60000 });
  await page.waitForFunction(() => window.__game && (window.__game.readyState === 'ready' || window.__game.readyState === 'error'), null, { timeout: timeoutMs });
  const state = await page.evaluate(() => ({ readyState: window.__game.readyState, error: window.__game.error }));
  if (state.readyState !== 'ready') { await browser.close(); server.close(); throw new Error('game boot failed: ' + state.error + '\n' + logs.join('\n')); }
  const close = async () => { await browser.close(); server.close(); };
  return { page, browser, server, url, logs, close };
}

export function parseArgs(argv) {
  const out = {}; for (let i = 0; i < argv.length; i++) { const a = argv[i]; if (a.startsWith('--')) { const k = a.slice(2); const v = argv[i + 1] !== undefined && !argv[i + 1].startsWith('--') ? argv[++i] : 'true'; out[k] = v; } }
  return out;
}

export function savePng(b64, file) { fs.mkdirSync(path.dirname(file), { recursive: true }); fs.writeFileSync(file, Buffer.from(b64, 'base64')); }
