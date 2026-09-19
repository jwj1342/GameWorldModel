#!/bin/bash
# 在登录节点上安装 Node 依赖和 Playwright 浏览器，打成一个 tar 放到 /project，作业里解到本地盘用。
# 计算节点访问不了 npm registry 和 Playwright CDN，所以这一步只能在登录节点做。用法：bash scripts/stage_node_deps.sh
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd)
source $REPO/scripts/setup_env.sh --no-venv
STAGE=${GWM_SCRATCH}/node_stage; mkdir -p "$STAGE" "$GWM_DEPS"
cp $REPO/package.json $REPO/package-lock.json "$STAGE/"
cd "$STAGE"
export npm_config_cache=${SCRATCH}/.npm-cache PLAYWRIGHT_BROWSERS_PATH=$STAGE/pw-browsers
npm ci --no-audit --no-fund
npx playwright install chromium
tar -cf "$GWM_DEPS/node-playwright-three.tar" node_modules pw-browsers package.json package-lock.json
sha256sum "$GWM_DEPS/node-playwright-three.tar" > "$GWM_DEPS/node-playwright-three.tar.sha256"
echo "staged $(du -h $GWM_DEPS/node-playwright-three.tar | cut -f1) -> $GWM_DEPS/node-playwright-three.tar"
