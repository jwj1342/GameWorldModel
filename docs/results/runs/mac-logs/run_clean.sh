#!/bin/bash
# 用干净的环境跑 scripts/bench_agent_run.sh：不把宿主会话的变量带给嵌套的 claude 和第三方接口。
# 用法：out/bench/logs/run_clean.sh <backend> <config> <label> [prepare 的额外参数...]
cd /Users/jwj/Code/GameWorldModel
LABEL=$3
exec env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$PATH" SHELL=/bin/zsh LANG=en_US.UTF-8 TMPDIR="$TMPDIR" TERM=dumb \
  /bin/bash scripts/bench_agent_run.sh "$@" > "out/bench/logs/$LABEL.log" 2>&1 < /dev/null
