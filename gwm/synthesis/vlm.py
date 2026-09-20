"""VLM client protocol + OpenAI-compatible implementation (vLLM). Images are sent as base64 data URLs; JSON output is schema-constrained."""
from __future__ import annotations
import base64, io, json, re, time
from pathlib import Path
from typing import Any, Protocol
from PIL import Image

class VLMClient(Protocol):
    def chat(self, system: str, user_text: str, images: list[str | Path] | None = None, json_schema: dict | None = None, temperature: float | None = None, max_tokens: int | None = None, seed: int | None = None) -> dict: ...

def _img_data_url(path: str | Path, max_side: int = 768) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB"); s = max_side / max(im.size)
        if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        buf = io.BytesIO(); im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()

def repair_json(text: str) -> str:
    """Cheap repairs for common model slips: trailing commas, missing commas between values, single quotes, NaN."""
    t = re.sub(r",\s*([}\]])", r"\1", text)                                  # trailing commas
    t = re.sub(r'([}\]"\d])\s*\n\s*(["{\[])', r"\1,\2", t)               # missing comma at line breaks
    t = re.sub(r'(\d)\s+(-?\d)', r"\1,\2", t)                              # missing comma between numbers
    t = re.sub(r'("\s*)\s+(")', r"\1,\2", t)                                # missing comma between strings
    t = re.sub(r"\bNaN\b|\bInfinity\b|-Infinity", "null", t)
    return t

def extract_json(text: str) -> Any:
    text = strip_think(text)
    for cand in (text, *(m.group(1) for m in re.finditer(r"```(?:json)?\s*(.*?)```", text, flags=re.S))):
        try: return json.loads(cand)
        except Exception: pass
    start = text.find("{"); end = text.rfind("}")
    if start < 0 or end <= start: raise ValueError("no JSON object found in model output")
    body = text[start:end + 1]
    try: return json.loads(body)
    except Exception: return json.loads(repair_json(body))

class OpenAICompatClient:
    """Talks to vLLM's OpenAI-compatible server. Endpoint is read from a file so a running Slurm job can publish it."""
    def __init__(self, cfg: dict, log_path: str | Path | None = None):
        from openai import OpenAI
        v = cfg["vlm"]; self.cfg = v
        self.kind = v.get("provider_kind", "vllm")          # vllm | openrouter | openai
        ep = v.get("endpoint") or Path(v["endpoint_file"]).read_text().strip()
        key = v.get("api_key") or "EMPTY"
        if v.get("api_key_file"):
            kp = Path(v["api_key_file"]);  kp = kp if kp.is_absolute() else Path(__file__).resolve().parents[2] / kp
            key = kp.read_text().strip()
        headers = {"HTTP-Referer": "https://github.com/jwj1342/GameWorldModel", "X-Title": "GameWorldModel"} if self.kind == "openrouter" else None
        self.client = OpenAI(base_url=ep, api_key=key, timeout=v.get("timeout_s", 600), max_retries=0, default_headers=headers)
        self.model = v.get("model") or self.client.models.list().data[0].id
        self.calls = 0; self.prompt_tokens = 0; self.completion_tokens = 0
        self.log_path = Path(log_path) if log_path else None
        self.max_image_side = v.get("max_image_side", 768)
        self.use_response_format = bool(v.get("json_schema_response_format", True))
        self.max_calls = int((cfg.get("budget") or {}).get("max_vlm_calls", 0)) or None
    def chat(self, system, user_text, images=None, json_schema=None, temperature=None, max_tokens=None, seed=None) -> dict:
        if self.max_calls and self.calls >= self.max_calls:
            raise RuntimeError(f"用掉了 {self.calls} 次模型调用，超过 budget.max_vlm_calls={self.max_calls}")
        content = [{"type": "image_url", "image_url": {"url": _img_data_url(p, self.max_image_side)}} for p in (images or [])]
        content.append({"type": "text", "text": user_text})
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": self.cfg["temperature"] if temperature is None else temperature,
                                  "top_p": self.cfg.get("top_p", 0.8), "max_tokens": max_tokens or self.cfg["max_tokens"]}
        if self.kind == "vllm":
            kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": bool(self.cfg.get("enable_thinking", False))}, "top_k": self.cfg.get("top_k", 20), "repetition_penalty": 1.0}
            if self.cfg.get("presence_penalty"): kwargs["presence_penalty"] = self.cfg["presence_penalty"]
        elif self.kind == "openrouter":
            extra: dict[str, Any] = {}
            if self.cfg.get("top_k"): extra["top_k"] = self.cfg["top_k"]
            if self.cfg.get("reasoning") is not None: extra["reasoning"] = self.cfg["reasoning"]   # e.g. {"effort": "low"} or {"exclude": true}
            if self.cfg.get("provider_routing"): extra["provider"] = self.cfg["provider_routing"]
            if extra: kwargs["extra_body"] = extra
        if seed is not None: kwargs["seed"] = seed
        if json_schema is not None:
            # belt and braces: providers differ in how strictly they honour response_format, so the schema is also spelled out in the prompt
            kwargs["messages"][0]["content"] += "\n\nRespond with a single JSON object only (no prose, no markdown fences), matching exactly this JSON Schema (same keys, same nesting):\n" + json.dumps(json_schema)[:8000]
            if self.use_response_format:
                kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": "out", "strict": True, "schema": json_schema}}
        last_err = None
        for attempt in range(self.cfg.get("max_retries", 3)):
            t0 = time.time()
            try:
                try:
                    r = self.client.chat.completions.create(**kwargs)
                except Exception as e:  # some providers reject response_format / extras: retry once without them, with the schema inlined
                    msg = str(e)
                    if ("response_format" in kwargs or "extra_body" in kwargs) and ("400" in msg or "invalid" in msg.lower() or "not supported" in msg.lower()):
                        kwargs.pop("extra_body", None)
                        kwargs.pop("response_format", None)
                        r = self.client.chat.completions.create(**kwargs)
                    else: raise
                text = r.choices[0].message.content or ""
                usage = getattr(r, "usage", None)
                self.calls += 1
                if usage: self.prompt_tokens += usage.prompt_tokens or 0; self.completion_tokens += usage.completion_tokens or 0
                parsed = extract_json(text) if json_schema is not None else None
                rec = {"time": time.strftime("%H:%M:%S"), "elapsed_s": round(time.time() - t0, 1), "n_images": len(images or []), "usage": usage.model_dump() if usage else None, "finish": r.choices[0].finish_reason, "text": text[:4000], "user_text": user_text[:1500], "system": system[:300]}
                if self.log_path: self.log_path.parent.mkdir(parents=True, exist_ok=True); self.log_path.open("a").write(json.dumps(rec, ensure_ascii=False) + "\n")
                return {"text": text, "json": parsed, "usage": rec["usage"], "finish": rec["finish"]}
            except Exception as e:  # network, timeout, bad JSON
                last_err = e
                raw = locals().get("text", "")
                if self.log_path: self.log_path.open("a").write(json.dumps({"time": time.strftime("%H:%M:%S"), "error": repr(e)[:500], "attempt": attempt, "raw": (raw or "")[:3000]}, ensure_ascii=False) + "\n")
                if isinstance(e, (ValueError, json.JSONDecodeError)) and attempt == 0:
                    kwargs["messages"][0]["content"] += "\n\nYour previous answer was not valid JSON. Output strictly valid JSON: commas between all elements, double quotes, no comments."
                time.sleep(min(2 ** attempt * 3, 30))
        raise RuntimeError(f"VLM call failed after retries: {last_err!r}")

def make_client(cfg: dict, log_path=None) -> VLMClient:
    prov = cfg["vlm"].get("provider", "openai_compat")
    if prov == "openai_compat": return OpenAICompatClient(cfg, log_path)
    if prov == "mock":
        from .mock_vlm import MockVLMClient; return MockVLMClient(cfg, log_path)
    raise NotImplementedError(prov)
