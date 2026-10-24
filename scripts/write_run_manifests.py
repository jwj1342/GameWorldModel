"""给 `rebuild_with_assets.py` 的产出目录补一份 `run.json`，好让 `eval.py` 能读。

为什么需要。`rebuild_with_assets.py` 写的是 `rebuild.json`，而 `eval.py` 读的是
`run.json`（它从里面取 clip 名去找真值，取 config_identity 做分组）。两者对不上，
于是整批评测会报一串 `FileNotFoundError(2)`——**而 GPU 那部分其实跑得好好的**，
只是最后一步读不到清单。踩过一次：一个三臂 A/B 的第一臂十二条全军覆没。

这个脚本只补清单，不碰程序。clip 名从目录名里取（约定是 `<前缀>_test_000NN`），
或者从 `program.json` 的 `meta.clip` 里读——后者更可靠，优先用它。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", action="append", required=True,
                    help="一个产出目录（里面有 program.json），可以给多次")
    ap.add_argument("--identity", required=True,
                    help="config_identity，评测按它分组，不同臂必须给不同的值")
    ap.add_argument("--overwrite", action="store_true",
                    help="已有 run.json 时也覆盖。默认不覆盖，免得把真的清单冲掉")
    a = ap.parse_args(argv)

    written = skipped = missing = 0
    for d in a.dir:
        run = Path(d)
        prog = run / "program.json"
        if not prog.is_file():
            print(f"  没有 program.json，跳过：{run}"); missing += 1; continue
        manifest = run / "run.json"
        if manifest.is_file() and not a.overwrite:
            skipped += 1; continue
        clip = None
        try:
            clip = ((json.loads(prog.read_text(encoding="utf-8")).get("meta") or {}).get("clip"))
        except (OSError, ValueError):
            pass
        if not clip:                       # 退回目录名，约定是 <前缀>_test_000NN
            name = run.name
            clip = name[name.index("test_"):] if "test_" in name else name
        manifest.write_text(json.dumps(
            {"clip": clip, "args": {"config": []}, "config_identity": a.identity},
            ensure_ascii=False) + "\n", encoding="utf-8")
        written += 1
    print(f"写了 {written} 份，已存在跳过 {skipped} 份，没有 program.json 的 {missing} 个")
    return 0 if written or skipped else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
