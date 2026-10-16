"""Load and merge YAML configs. Usage: cfg = load_config(['configs/default.yaml', 'configs/vulcan.yaml', 'configs/ablations/x.yaml'])"""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any
import yaml

REPO = Path(__file__).resolve().parents[1]

def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out

def _expand(v: Any) -> Any:
    if isinstance(v, str): return os.path.expandvars(v)
    if isinstance(v, dict): return {k: _expand(x) for k, x in v.items()}
    if isinstance(v, list): return [_expand(x) for x in v]
    return v

def site_name() -> str:
    """Which machine-specific config to use. GWM_SITE wins; otherwise Vulcan is detected by its Slurm/CVMFS environment."""
    env = os.environ.get("GWM_SITE")
    if env: return env
    on_cluster = os.environ.get("CC_CLUSTER") or Path("/cvmfs/soft.computecanada.ca").exists()
    return "vulcan" if on_cluster else "local"

def load_config(files: list[str | Path] | None = None, overrides: dict | None = None) -> dict:
    files = files or [REPO / "configs" / "default.yaml", REPO / "configs" / f"{site_name()}.yaml"]
    cfg: dict = {}
    for f in files:
        p = Path(f)
        if not p.is_absolute(): p = REPO / p
        if p.exists(): cfg = _merge(cfg, yaml.safe_load(p.read_text(encoding="utf-8")) or {})
    if overrides: cfg = _merge(cfg, overrides)
    cfg = _expand(cfg)
    for k, v in (cfg.get("paths") or {}).items():
        if isinstance(v, str) and not Path(v).is_absolute(): cfg["paths"][k] = str(REPO / v)
    cfg.setdefault("site", site_name())
    return cfg


SECRET_KEYS = re.compile(r"key|token|secret|password|authorization|credential", re.I)
# Exact names only: unknown spellings fail closed instead of exposing credentials.
NOT_SECRET = frozenset({"keyframes", "keyframe_max_side", "detection_keyframes", "max_tokens",
                        "hazard_keywords"})

def redact(value: Any) -> Any:
    """把配置里像密钥的字段换成占位符。run.json 会被 collect_results.sh 复制进入库的 docs/results/，
    所以写进去之前必须过一遍，别让人不小心把 key 写进 config 覆盖文件就提交上去。
    凭据文件路径也隐藏，与配置身份摘要共用规则；不会读取凭据文件。"""
    if isinstance(value, dict):
        return {k: ("<redacted>" if isinstance(k, str) and SECRET_KEYS.search(k) and k not in NOT_SECRET
                    else redact(v)) for k, v in value.items()}
    if isinstance(value, list): return [redact(v) for v in value]
    return value
