#!/usr/bin/env python
"""LoRA 微调 Qwen3-VL：两帧 + 一句话 + 已知的程序部分 → objects。

预实验要回答的只有一个问题：这个映射能不能被一个 8B 的开源多模态模型学到。
所以训练循环写得最直白：batch 1、梯度累积、bf16、梯度检查点、每轮在留出集上算一次 loss、存一次适配器。
多卡用 torchrun 起数据并行：每张卡每步一条，等效批大小是 卡数 × accum。

用法:
  python scripts/statediff/train_lora.py --samples SET/samples.jsonl --model $GWM_WEIGHTS/qwen3-vl-8b-instruct \
      --out $GWM_SCRATCH/statediff/lora1 --epochs 4
  torchrun --nproc_per_node 4 scripts/statediff/train_lora.py ... --accum 2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.statediff.common import load_samples                            # noqa: E402
from scripts.statediff.vlm_io import encode_pair, target_for, to_device     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--train-set", default="train")
    ap.add_argument("--eval-set", default="holdout_in")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=7168)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条训练（冒烟）")
    ap.add_argument("--eval-limit", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init-adapter", default="", help="先把这个适配器并进底座再挂新的 LoRA（接着已训好的 SFT 往下训）")
    ap.add_argument("--extra-samples", default="", help="另一份样本（比如无标注池），目标文本取自 --extra-targets 而不是真值程序")
    ap.add_argument("--extra-targets", default="", help="预测格式的 jsonl（id、text），当伪标签")
    ap.add_argument("--extra-min-score", type=float, default=0.0, help="伪标签按 search_score 过滤，没有这个字段的不过滤")
    ap.add_argument("--train-limit-after", type=int, default=0, help="真值训练样本只留前 N 条（接着训时当回放，防遗忘）")
    a = ap.parse_args()

    import os

    import torch
    import torch.distributed as dist
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8") if rank == 0 else None

    def emit(**rec):
        if rank != 0:
            return
        rec["time"] = time.strftime("%H:%M:%S")
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()
        print(json.dumps(rec, ensure_ascii=False), flush=True)

    train = load_samples(a.samples, [a.train_set])
    evals = load_samples(a.samples, [a.eval_set])[: a.eval_limit]
    if a.limit:
        train = train[: a.limit]
    if a.train_limit_after:
        train = random.Random(a.seed).sample(train, min(a.train_limit_after, len(train)))
    pseudo = {}
    if a.extra_samples:
        for line in Path(a.extra_targets).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("json") is not None and (r.get("search_score") is None or r["search_score"] >= a.extra_min_score):
                    pseudo[r["id"]] = r["text"]
        train += [s for s in load_samples(a.extra_samples) if s["id"] in pseudo]
    if not train:
        print("没有训练样本", file=sys.stderr)
        return 1
    answers = {s["id"]: pseudo.get(s["id"]) or target_for(s) for s in train + evals}

    device = f"cuda:{torch.cuda.current_device()}"
    processor = AutoProcessor.from_pretrained(a.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter).merge_and_unload()
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    model = get_peft_model(model, lora)
    core = model
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    emit(event="setup", train=len(train), eval=len(evals), trainable_params=n_train, all_params=n_all,
         model=a.model, rank=a.rank, lr=a.lr, accum=a.accum, epochs=a.epochs, world=world,
         pseudo=len(pseudo), init_adapter=a.init_adapter)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_rank = len(train) // world
    updates_total = max(1, math.ceil(per_rank * a.epochs / a.accum))
    warm = max(1, min(20, updates_total // 10))

    def lr_at(step: int) -> float:
        if step < warm:
            return a.lr * (step + 1) / warm
        p = (step - warm) / max(1, updates_total - warm)
        return a.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))

    def eval_loss() -> float | None:
        if not evals or rank != 0:
            return None
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for s in evals:
                enc, labels, _ = encode_pair(processor, s, answers[s["id"]], a.max_len)
                outp = core(**to_device(enc, device), labels=labels.to(device))
                tot += float(outp.loss)
                n += 1
        model.train()
        return tot / max(n, 1)

    step, seen, t0 = 0, 0, time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(train)
        random.Random(a.seed + ep).shuffle(order)
        order = order[rank: per_rank * world: world]       # 每张卡拿同样多条，步数对齐
        run_loss, run_n, truncated = 0.0, 0, 0
        for i, s in enumerate(order, 1):
            enc, labels, info = encode_pair(processor, s, answers[s["id"]], a.max_len)
            truncated += int(info["truncated"])
            sync = i % a.accum == 0 or i == len(order)
            with (model.no_sync() if world > 1 and not sync else contextlib.nullcontext()):
                outp = model(**to_device(enc, device), labels=labels.to(device))
                (outp.loss / a.accum).backward()
            run_loss += float(outp.loss)
            run_n += 1
            seen += 1
            if sync:
                for g in opt.param_groups:
                    g["lr"] = lr_at(step)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if i % 20 == 0 or i == len(order):
                emit(event="train", epoch=ep, seen=seen * world, step=step, loss=round(run_loss / run_n, 4),
                     lr=round(lr_at(step), 7), tokens=info["total_tokens"], prompt_tokens=info["prompt_tokens"],
                     truncated=truncated, elapsed_s=round(time.time() - t0),
                     gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
                run_loss, run_n = 0.0, 0
        ev = eval_loss()
        emit(event="epoch_end", epoch=ep, eval_loss=None if ev is None else round(ev, 4), elapsed_s=round(time.time() - t0))
        if rank == 0:
            core.save_pretrained(str(out / f"epoch{ep}"))
        if world > 1:
            dist.barrier()
    if rank == 0:
        core.save_pretrained(str(out / "final"))
    emit(event="done", elapsed_s=round(time.time() - t0), adapter=str(out / "final"))
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
