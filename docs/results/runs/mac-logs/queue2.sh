#!/bin/bash
# 第二版排队：Kimi 的 5 小时额度用完时（429）不往下跳，等额度恢复后重跑同一项。
# 其他原因的失败记下来跳到下一项；连续两次这类失败就停。
cd /Users/jwj/Code/GameWorldModel
Q=out/bench/logs/queue.log
say() { echo "$(date '+%m-%d %H:%M:%S') $*" >> "$Q"; }
clean() { env -i HOME="$HOME" USER="$USER" LOGNAME="$USER" PATH="$PATH" SHELL=/bin/zsh LANG=en_US.UTF-8 TMPDIR="$TMPDIR" TERM=dumb "$@"; }
quota_ok() {  # 发一条最小请求，看 Kimi 还收不收
  local out
  out=$(cd "$TMPDIR" && clean /bin/bash -c 'ANTHROPIC_BASE_URL=https://api.kimi.com/coding/ ANTHROPIC_API_KEY=$(cat "$HOME/.config/gwm-bench/kimi_api_key") claude -p "Reply with the single word: ok" --model kimi-for-coding --output-format json --disallowedTools WebSearch WebFetch < /dev/null' 2>/dev/null)
  echo "$out" | grep -q '"is_error":false'
}
wait_quota() {
  local waited=0
  until quota_ok; do
    [ $waited -eq 0 ] && say "WAIT 额度未恢复，每 15 分钟探一次"
    waited=1; sleep 900
  done
  [ $waited -eq 1 ] && say "QUOTA OK"
}
hit_quota() {  # 最近一个该 label 的工作区里，运行记录是不是 429
  local w; w=$(ls -d out/bench/*/workspaces/*-"$1" 2>/dev/null | tail -1)
  [ -n "$w" ] && [ -f "$w/agent_run.json" ] && grep -q -E "429|usage limit" "$w/agent_run.json"
}
fails=0
run() {  # run <config> <label> [extra...]
  local cfg=$1 label=$2 try; shift 2
  for try in 1 2 3 4; do
    wait_quota
    say "START $label（第 $try 次）"
    out/bench/logs/run_clean.sh kimi "$cfg" "$label" "$@"
    if grep -q "^DONE $label" "out/bench/logs/$label.log"; then say "DONE $label"; fails=0; return; fi
    if hit_quota "$label"; then say "QUOTA $label 中途撞上额度上限，等恢复后重跑"; sleep 600; continue; fi
    say "FAILED $label（见 out/bench/logs/$label.log）"; fails=$((fails+1))
    if [ $fails -ge 2 ]; then say "STOP 连续两次失败"; exit 2; fi
    return
  done
  say "GIVE UP $label 四次都撞上额度上限"
}
run configs/bench/room.yaml kimi-dense --inputs dense
run configs/bench/room.yaml kimi-photos --inputs photos
run configs/bench/room.yaml kimi-score --score-tool
run configs/bench/room.yaml kimi-nomanifest --no-manifest
run configs/bench/counter.yaml kimi-counter
say "QUEUE FINISHED"
