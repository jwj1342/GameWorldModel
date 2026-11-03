#!/usr/bin/env python
"""用本地 Qwen3-VL（可带 LoRA 适配器）对样本逐条生成 objects，写成和提示基线同一格式的预测文件。

用法:
  python scripts/statediff/infer.py --samples SET/samples.jsonl --sets holdout_in,holdout_out \
      --model $GWM_WEIGHTS/qwen3-vl-8b-instruct --adapter LORA/final --out PRED.jsonl
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

from gwm.synthesis.vlm import extract_json                      # noqa: E402
from scripts.statediff.common import load_samples              # noqa: E402
from scripts.statediff.vlm_io import encode_prompt, to_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--sets", default="holdout_in,holdout_out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--per-set-limit", type=int, default=0)
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

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
    print(f"{len(todo)} to generate ({len(done)} already in {out})", flush=True)
    if not todo:
        return 0

    processor = AutoProcessor.from_pretrained(a.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload()
    model.to("cuda").eval()

    with out.open("a", encoding="utf-8") as f:
        for i, s in enumerate(todo, 1):
            t0 = time.time()
            enc, plen = encode_prompt(processor, s)
            with torch.no_grad():
                gen = model.generate(**to_device(enc, "cuda"), max_new_tokens=a.max_new_tokens, do_sample=False)
            new = gen[0, plen:]
            text = processor.batch_decode([new], skip_special_tokens=True)[0]
            try:
                parsed = extract_json(text)
            except Exception:       # 不是 JSON 也要记下来，评测把它算成无效
                parsed = None
            rec = {"id": s["id"], "set": s["set"], "text": text, "json": parsed, "prompt_tokens": plen,
                   "new_tokens": int(new.shape[0]), "gen_s": round(time.time() - t0, 1),
                   "source": "local:" + (a.adapter or a.model)}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 5 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}  last {rec['new_tokens']} tok in {rec['gen_s']}s  json={'ok' if parsed is not None else 'none'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
