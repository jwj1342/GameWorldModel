#!/bin/bash
# 在有 node_modules 的目录里运行 harness 脚本（ESM 解析不看 NODE_PATH）。
# 用法: scripts/node_harness.sh render.mjs --game DIR ...   需要 GWM_NODE_ROOT（含 node_modules 与 pw-browsers）
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${GWM_NODE_ROOT:-${SLURM_TMPDIR:-/tmp}/node}
if [ ! -d "$ROOT/node_modules" ]; then
  mkdir -p "$ROOT"; tar -xf "${GWM_DEPS:-/project/aip-zhouyang/jwj/GameWorldModel/deps}/node-playwright-three.tar" -C "$ROOT"
fi
export PLAYWRIGHT_BROWSERS_PATH=$ROOT/pw-browsers GWM_NODE_MODULES=$ROOT/node_modules
mkdir -p "$ROOT/harness"; cp -u "$REPO"/harness/*.mjs "$REPO"/harness/package.json "$ROOT/harness/"
script=$1; shift
exec node "$ROOT/harness/$script" "$@"
