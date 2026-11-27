"""本地 Qwen3-VL 的输入编码，训练和推理共用，保证两边喂的是同一份提示。"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.statediff.common import context_of, read_json, system_prompt, user_text  # noqa: E402


def messages_for(sample: dict, answer: str | None = None) -> list[dict]:
    context = context_of(read_json(sample["program"]))
    msgs = [{"role": "system", "content": system_prompt()},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"},
                                         {"type": "text", "text": user_text(context, sample["text"])}]}]
    if answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return msgs


def images_for(sample: dict) -> list[Image.Image]:
    return [Image.open(sample["img0"]).convert("RGB"), Image.open(sample["img1"]).convert("RGB")]


def encode_prompt(processor, sample: dict):
    """只有提示（不含答案），给推理用。返回 (inputs, prompt_len)。"""
    text = processor.apply_chat_template(messages_for(sample), tokenize=False, add_generation_prompt=True)
    enc = processor(text=[text], images=images_for(sample), return_tensors="pt")
    return enc, int(enc["input_ids"].shape[1])


def encode_pair(processor, sample: dict, answer: str, max_len: int):
    """提示 + 答案，给训练用。labels 里提示部分是 -100。返回 (inputs, labels, info)。"""
    import torch
    prompt_text = processor.apply_chat_template(messages_for(sample), tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages_for(sample, answer), tokenize=False, add_generation_prompt=False)
    if not full_text.startswith(prompt_text):      # 模板不按常理出牌时退回手拼，保证前缀一致
        full_text = prompt_text + answer + "<|im_end|>\n"
    imgs = images_for(sample)
    enc_prompt = processor(text=[prompt_text], images=imgs, return_tensors="pt")
    enc = processor(text=[full_text], images=imgs, return_tensors="pt")
    plen = int(enc_prompt["input_ids"].shape[1])
    labels = enc["input_ids"].clone()
    labels[:, :plen] = -100
    info = {"prompt_tokens": plen, "total_tokens": int(enc["input_ids"].shape[1]), "truncated": False}
    if info["total_tokens"] > max_len:
        info["truncated"] = True
        for k in ("input_ids", "attention_mask"):
            enc[k] = enc[k][:, :max_len]
        labels = labels[:, :max_len]
    return enc, labels, info


def to_device(enc, device):
    import torch
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}
