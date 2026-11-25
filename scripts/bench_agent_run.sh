#!/bin/bash
# 两轴评测的一次完整运行：准备工作区 -> 智能体写场景 -> eval（纯色）-> 快照轨迹 -> 把要看的结果拷进 docs/results/runs/。
#
#   scripts/bench_agent_run.sh <claude|kimi> <config> <label> [agent prepare 的额外参数...]
#
#   scripts/bench_agent_run.sh kimi configs/bench/room.yaml kimi-var1
#   scripts/bench_agent_run.sh kimi configs/bench/room.yaml kimi-dense --inputs dense
#
# claude：用 claude 命令行当前的登录（订阅额度），模型按配置里的 agent.model。
# kimi：同一个 claude 命令行，经 Kimi 兼容 Anthropic 的接口驱动 kimi-for-coding，不占 Claude 的额度。
#       key 放在 ~/.config/gwm-bench/kimi_api_key（chmod 600），不进 git，脚本不打印它。
# 在仓库根目录跑。Python 默认用 venv/bin/python，可以用 PYTHON= 覆盖。
set -euo pipefail
BACKEND=$1; CFG=$2; LABEL=$3; shift 3
PY=${PYTHON:-venv/bin/python}
SCENE=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['scene'])" "$CFG")
W=out/bench/$SCENE/workspaces/$(date +%Y%m%d-%H%M%S)-$LABEL
echo "workspace $W"
"$PY" -m gwm.bench.agent prepare --config "$CFG" --workspace "$W" "$@"
case "$BACKEND" in
  claude) "$PY" -m gwm.bench.agent run --config "$CFG" --workspace "$W" ;;
  kimi)   ANTHROPIC_BASE_URL=https://api.kimi.com/coding/ \
          ANTHROPIC_API_KEY=$(cat "$HOME/.config/gwm-bench/kimi_api_key") \
          "$PY" -m gwm.bench.agent run --config "$CFG" --workspace "$W" --model kimi-for-coding ;;
  *) echo "backend 只能是 claude 或 kimi" >&2; exit 1 ;;
esac
"$PY" -m gwm.bench.run eval --config "$CFG" --cand "$W" --label "$LABEL" --modes flat
"$PY" -m gwm.bench.run trajectory --config "$CFG" --workspace "$W" --label "$LABEL"

# 要看的东西拷一份进仓库，提交推上去就能在别的机器上分析。渲染图和贴图不拷，太大。
RUN=$(ls -d out/bench/$SCENE/runs/*-"$LABEL" | tail -1)
TRAJ=$(ls -d out/bench/$SCENE/trajectories/*-"$LABEL" | tail -1)
DEST=docs/results/runs/$SCENE-$LABEL
mkdir -p "$DEST"
cp "$W/scene.js" "$W/materials.json" "$W/agent_run.json" "$DEST/"
cp "$RUN"/results.json "$RUN"/fidelity_flat.json "$RUN"/geometry.json "$RUN"/interact.json "$DEST/"
# 连续交互只有配置里写了 sequence 的场景才有（room 有，counter 没有）
if ls "$RUN"/sequence_*.json >/dev/null 2>&1; then cp "$RUN"/sequence_*.json "$DEST/"; fi
cp "$TRAJ/trajectory.json" "$DEST/"
echo "{\"type\":\"module\"}" > "$DEST/package.json"
echo "DONE $LABEL -> $DEST"
