#!/bin/bash
# 排队：先等 kimi-var2 结束。它打印了 DONE 才往下串行跑；连续两次没有 DONE 就停，免得限流时空转一晚。
cd /Users/jwj/Code/GameWorldModel
Q=out/bench/logs/queue.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$Q"; }
while pgrep -f "bench_agent_run.sh kimi configs/bench/room.yaml kimi-var2" >/dev/null; do sleep 30; done
if ! grep -q "^DONE kimi-var2" out/bench/logs/kimi-var2.log; then say "GATE FAILED kimi-var2 没有打印 DONE，不往下跑"; exit 1; fi
say "GATE OK kimi-var2"
fails=0
run() {  # run <config> <label> [extra...]
  local cfg=$1 label=$2; shift 2
  say "START $label"
  out/bench/logs/run_clean.sh kimi "$cfg" "$label" "$@"
  if grep -q "^DONE $label" "out/bench/logs/$label.log"; then say "DONE $label"; fails=0
  else say "FAILED $label（见 out/bench/logs/$label.log）"; fails=$((fails+1)); fi
  if [ $fails -ge 2 ]; then say "STOP 连续两次失败"; exit 2; fi
}
run configs/bench/room.yaml kimi-var3
run configs/bench/room.yaml kimi-dense --inputs dense
run configs/bench/room.yaml kimi-photos --inputs photos
run configs/bench/room.yaml kimi-score --score-tool
run configs/bench/room.yaml kimi-nomanifest --no-manifest
run configs/bench/counter.yaml kimi-counter
say "QUEUE FINISHED"
