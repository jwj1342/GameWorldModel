#!/bin/bash
# 在有 node_modules 的目录里运行 harness 脚本（ESM 解析不看 NODE_PATH）。
# 用法: scripts/node_harness.sh render.mjs --game DIR ...   需要 GWM_NODE_ROOT（含 node_modules 与 pw-browsers）
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
# 普通机器上 npm install 之后仓库里就有 node_modules，直接用；集群上没有，从打好的 tar 解到本地盘
if [ -d "$REPO/node_modules" ] && [ -z "${GWM_NODE_ROOT:-}" ]; then
  script=$1; shift
  exec node "$REPO/harness/$script" "$@"
fi
ROOT=${GWM_NODE_ROOT:-${SLURM_TMPDIR:-/tmp}/node}
if [ ! -d "$ROOT/node_modules" ]; then
  TAR="${GWM_DEPS:-/project/aip-zhouyang/jwj/GameWorldModel/deps}/node-playwright-three.tar"
  [ -f "$TAR" ] || { echo "找不到 node_modules，也找不到 $TAR。在仓库里执行 npm install 就行。" >&2; exit 1; }
  mkdir -p "$ROOT"; tar -xf "$TAR" -C "$ROOT"
fi
export PLAYWRIGHT_BROWSERS_PATH=$ROOT/pw-browsers GWM_NODE_MODULES=$ROOT/node_modules
mkdir -p "$ROOT/harness"; cp -u "$REPO"/harness/*.mjs "$REPO"/harness/package.json "$ROOT/harness/"
script=$1; shift
exec node "$ROOT/harness/$script" "$@"
