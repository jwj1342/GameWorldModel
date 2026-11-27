#!/usr/bin/env python
"""在 SFT 适配器上做离线 DPO：偏好对来自内核执行打分（build_pairs.py）。

做法：把 SFT 适配器并进底座当参考模型，再挂一个新的 LoRA 当策略；参考模型的对数概率用关掉适配器的同一个模型算。
每个偏好对把 chosen 和 rejected 拼成批大小 2 做一次前向，DDP 下每步只有一次前向一次反向。
可选在 chosen 上加一项 NLL（--nll-weight），防止只推开 rejected 时把格式也推坏。

用法:
  torchrun --nproc_per_node 4 scripts/statediff/train_dpo.py --samples SET/samples.jsonl --pairs pairs.jsonl \
      --model $GWM_WEIGHTS/qwen3-vl-8b-instruct --sft LORA/final --out DPO --epochs 2
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.statediff.common import load_samples                     # noqa: E402
from scripts.statediff.vlm_io import images_for, messages_for, to_device  # noqa: E402


def encode_two(processor, sample: dict, chosen: str, rejected: str, max_len: int):
    prompt = processor.apply_chat_template(messages_for(sample), tokenize=False, add_generation_prompt=True)
    fulls = []
    for ans in (chosen, rejected):
        f = processor.apply_chat_template(messages_for(sample, ans), tokenize=False, add_generation_prompt=False)
        fulls.append(f if f.startswith(prompt) else prompt + ans + "<|im_end|>\n")
    imgs = images_for(sample)
    plen = int(processor(text=[prompt], images=imgs, return_tensors="pt")["input_ids"].shape[1])
    enc = processor(text=fulls, images=imgs + imgs, return_tensors="pt", padding=True)
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    labels[enc["attention_mask"] == 0] = -100
    if labels.shape[1] > max_len:
        for k in ("input_ids", "attention_mask"):
            enc[k] = enc[k][:, :max_len]
        labels = labels[:, :max_len]
    return enc, labels


def seq_logps(logits, labels):
    """每行答案部分的对数概率之和，以及答案 token 数。"""
    import torch
    lg, lb = logits[:, :-1], labels[:, 1:]
    mask = lb != -100
    lp = torch.gather(torch.log_softmax(lg.float(), -1), 2, lb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(-1), mask.sum(-1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--samples", required=True, help="可给多份，用逗号分开")
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--sft", required=True, help="SFT 适配器目录，并进底座当参考")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--nll-weight", type=float, default=0.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=7168)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    world, rank = int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = f"cuda:{torch.cuda.current_device()}"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = (out / "train_log.jsonl").open("a", encoding="utf-8") if rank == 0 else None

    def emit(**rec):
        if rank == 0:
            rec["time"] = time.strftime("%H:%M:%S")
            log.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log.flush()
            print(json.dumps(rec, ensure_ascii=False), flush=True)

    samples = {s["id"]: s for p in a.samples.split(",") for s in load_samples(p)}
    pairs = [json.loads(l) for l in Path(a.pairs).read_text(encoding="utf-8").splitlines() if l.strip()]
    pairs = [p for p in pairs if p["id"] in samples]

    processor = AutoProcessor.from_pretrained(a.model)
    processor.tokenizer.padding_side = "right"
    base = Qwen3VLForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa")
    base = PeftModel.from_pretrained(base, a.sft).merge_and_unload()
    base.to(device)
    base.config.use_cache = False
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    lora = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    core = get_peft_model(base, lora)
    model = torch.nn.parallel.DistributedDataParallel(core, device_ids=[torch.cuda.current_device()]) if world > 1 else core
    params = [p for p in core.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    per_rank = len(pairs) // world
    updates_total = max(1, math.ceil(per_rank * a.epochs / a.accum))
    emit(event="setup", pairs=len(pairs), per_rank=per_rank, world=world, beta=a.beta, lr=a.lr, nll_weight=a.nll_weight,
         epochs=a.epochs, accum=a.accum, updates=updates_total)

    step, t0 = 0, time.time()
    model.train()
    for ep in range(a.epochs):
        order = list(pairs)
        random.Random(a.seed + ep).shuffle(order)
        order = order[rank: per_rank * world: world]
        acc = {"loss": 0.0, "margin": 0.0, "win": 0.0, "n": 0}
        for i, p in enumerate(order, 1):
            enc, labels = encode_two(processor, samples[p["id"]], p["chosen"], p["rejected"], a.max_len)
            enc, labels = to_device(enc, device), labels.to(device)
            with torch.no_grad(), core.disable_adapter():
                ref, _ = seq_logps(core(**enc).logits, labels)
            sync = i % a.accum == 0 or i == len(order)
            with (model.no_sync() if world > 1 and not sync else contextlib.nullcontext()):
                pol, ntok = seq_logps(model(**enc).logits, labels)
                margin = (pol[0] - ref[0]) - (pol[1] - ref[1])
                loss = -F.logsigmoid(a.beta * margin) + a.nll_weight * (-pol[0] / ntok[0])
                (loss / a.accum).backward()
            acc["loss"] += float(loss)
            acc["margin"] += float(margin)
            acc["win"] += float(margin > 0)
            acc["n"] += 1
            if sync:
                lr = a.lr * min(1.0, (step + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / updates_total)))
                for g in opt.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
            if i % 10 == 0 or i == len(order):
                emit(event="train", epoch=ep, seen=i * world, step=step, loss=round(acc["loss"] / acc["n"], 4),
                     margin=round(acc["margin"] / acc["n"], 3), win=round(acc["win"] / acc["n"], 3),
                     elapsed_s=round(time.time() - t0), gpu_mem_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 1))
                acc = {"loss": 0.0, "margin": 0.0, "win": 0.0, "n": 0}
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
