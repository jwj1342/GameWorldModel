#!/usr/bin/env python3
"""起一个本地服务器把游戏跑起来。浏览器不允许 file:// 加载 ES 模块，所以不能直接双击 index.html。

    python3 serve.py          然后打开 http://localhost:8000/

WASD 或方向键移动，空格跳，R 在回放和游玩之间切换。"""
import http.server, socketserver, os, sys, webbrowser

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "game"))

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()
    def log_message(self, *a): pass

with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
    url = f"http://localhost:{PORT}/"
    print(f"游戏在 {url} ，按 Ctrl+C 退出")
    try: webbrowser.open(url)
    except Exception: pass
    httpd.serve_forever()
