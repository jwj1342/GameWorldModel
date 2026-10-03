// 给人用的本地静态服务: node harness/serve.mjs <目录> [端口]
import { serveDir } from './common.mjs';

const root = process.argv[2] ?? '.';
const port = Number(process.argv[3] ?? 8080);
const { url } = await serveDir(root, port);
console.log(`serving ${root} at ${url}/`);
