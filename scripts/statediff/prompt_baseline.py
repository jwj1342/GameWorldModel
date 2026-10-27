#!/usr/bin/env python
"""提示现成模型（OpenRouter / 自托管）做同一个任务，写成和 infer.py 同一格式的预测文件。

默认模型取 configs 里 vlm.model（GLM-4.6V）。要换模型：--model z-ai/glm-4.6v 或 --config configs/vlm/openrouter_qwen122b.yaml。
密钥按 configs 的 vlm.api_key_file 读；作业 worktree 里没有 .secrets 时用 --key-file 指到主仓库那份。

用法:
  python scripts/statediff/prompt_baseline.py --samples SET/samples.jsonl --sets holdout_in,holdout_out --out PRED.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from gwm.config import load_config                                             # noqa: E402
from gwm.synthesis.vlm import make_client                                      # noqa: E402
from scripts.statediff.common import (context_of, load_samples, objects_schema,  # noqa: E402
                                      read_json, system_prompt, user_text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sets", default="holdout_in,holdout_out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", action="append", default=[], help="额外叠加的 yaml，比如 configs/vlm/openrouter_qwen122b.yaml")
    ap.add_argument("--model", default="")
    ap.add_argument("--key-file", default="")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--per-set-limit", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()

    files = [REPO / "configs" / "default.yaml", REPO / "configs" / "vulcan.yaml", *a.config]
    cfg = load_config(files=[f for f in files if Path(f).exists()])
    cfg["vlm"]["max_tokens"] = a.max_tokens
    cfg["vlm"]["max_image_side"] = 1280            # 默认 768 会把 1280 宽的帧缩一半
    if a.model:
        cfg["vlm"]["model"] = a.model
    if a.key_file:
        cfg["vlm"]["api_key_file"] = a.key_file
    cfg.setdefault("budget", {})["max_vlm_calls"] = 0

    samples = []
    for set_name in a.sets.split(","):
        rows = load_samples(a.samples, [set_name])
        samples += rows[: a.per_set_limit] if a.per_set_limit else rows
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["id"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    todo = [s for s in samples if s["id"] not in done]
    print(f"{len(todo)} to query with {cfg['vlm']['model']} ({len(done)} already in {out})", flush=True)
    if not todo:
        return 0
    client = make_client(cfg, log_path=str(out) + ".calls.jsonl")
    schema = objects_schema()
    system = system_prompt()
    failures = 0
    with out.open("a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            t0 = time.time()
            context = context_of(read_json(s["program"]))
            try:
                r = client.chat(system, user_text(context, s["text"]), images=[s["img0"], s["img1"]],
                                json_schema=schema, temperature=a.temperature)
                rec = {"id": s["id"], "set": s["set"], "text": r["text"], "json": r["json"], "usage": r["usage"],
                       "finish": r["finish"]}
            except Exception as e:          # 网络、超时、连修都修不好的 JSON：记下来当无效
                failures += 1
                rec = {"id": s["id"], "set": s["set"], "text": "", "json": None, "error": repr(e)[:300]}
            rec.update(gen_s=round(time.time() - t0, 1), source="prompt:" + cfg["vlm"]["model"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  {rec['gen_s']}s  json={'ok' if rec['json'] is not None else 'none'}  failures={failures}", flush=True)
    print(json.dumps({"calls": client.calls, "prompt_tokens": client.prompt_tokens,
                      "completion_tokens": client.completion_tokens, "failures": failures}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
