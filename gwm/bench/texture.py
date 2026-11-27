"""把 materials.json 里的一条条文字描述，变成可以平铺的贴图。

评测 demo 里，场景是智能体写出来的 three.js 代码；每个表面「长什么样」这种高熵的信息
不写进代码，只写成一句提示词加一个种子。这个模块负责把它们落成图：
`名字 -> {"prompt": ..., "seed": ...}` 进，`<名字>.png` 出，外加一份 `textures.json`
记下每张图到底是怎么来的。

三件要紧的事：

* **可平铺**。生成之前把 UNet 和 VAE 里每一个 `Conv2d` 都换成环绕补边
  （`padding_mode = "circular"`）。这样整个网络在环面上是平移等变的，出来的图
  左右、上下首尾相接。`seam_error` 用纯 numpy 量这件事，结果也写进日志，不靠目测。
* **可复现**。每张图用条目自己的种子在 CPU 上建 `torch.Generator`
  （MPS 上的生成器和 CPU 不是同一条随机流，放在 CPU 上才和设备无关）。
  同一台机器上，同样的 (提示词, 种子, 模型) 得到同样的图。
* **重跑便宜**。PNG 还在、日志里记的提示词、种子、模型、尺寸、步数都对得上，就跳过；
  全部跳过时连 torch 都不会导入。

模型只在本机跑，权重放在仓库的 `models/hf/` 下（已被 git 忽略），不走任何云端接口。
默认模型是 `stabilityai/sd-turbo`（SD 2.1 的蒸馏版，512 是它的原生分辨率），许可是
Stability AI Community License：研究和非商用免费，商用有营收门槛，细则见模型仓库的 LICENSE.md。

为什么不是更大的 `stabilityai/sdxl-turbo`：在这台机器上实测过，SDXL-Turbo 一旦全程环绕补边，
画面就散了（木地板变成卡通拼花，墙面变成龟裂的石块，布料变成鹅卵石）；它的布局明显依赖
零补边带来的边界信息。SD-Turbo 没有这个问题，环绕补边前后画质基本不变，而且更快。
`stable-diffusion-v1-5` 环绕补边也成立，但一张要二十秒上下，而且经常跑题。

命令行：

    venv/bin/python -m gwm.bench.texture --materials PATH/materials.json --out DIR \
        [--model ID] [--size 512] [--steps N] [--device mps]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np

REPO = Path(__file__).resolve().parents[2]
# 权重目录。models/ 在 .gitignore 里；不用 ~/.cache/huggingface，免得权重散在仓库外面。
MODELS_DIR = REPO / "models" / "hf"

DEFAULT_MODEL = "stabilityai/sd-turbo"

# 提示词模板。目的是要一块「平的材质样片」，而不是一张拍了某个物体的照片：
# 正视、均匀光、没有投影。蒸馏模型（turbo）不用引导，负面提示词不起作用，
# 所以「不要什么」这层意思只能靠正面描述里的 flat / even / top-down 来表达。
# 试过换成「close-up photograph」一类的写法：细节更像照片，但会带上透视和景深，不平了。
PROMPT_TEMPLATE = ("seamless tileable texture of {prompt}, flat even lighting, "
                   "top-down orthographic view, albedo map, no shadows, no objects")
# 只在引导系数大于 1 的模型上生效（见 model_defaults）。
NEGATIVE_PROMPT = ("shadows, perspective, vignette, 3d render, object, furniture, room, "
                   "text, watermark, logo, frame, border, blurry")

NAME_PATTERN = re.compile(r"^[a-z0-9_]+$")
LOG_NAME = "textures.json"
# 决定「这张图要不要重画」的字段。模型版本号不在里面：要先解析模型才知道，
# 而跳过检查必须在加载模型之前就能做完。
IDENTITY_KEYS = ("full_prompt", "negative_prompt", "seed", "model", "size", "steps", "guidance")


class TextureError(ValueError):
    """materials 里的内容不合规矩。消息里带着是哪一条、错在哪。"""


class Renderer(Protocol):
    """真正出图的那个东西。测试里换成假的，就不用加载模型。"""
    device: str
    revision: str | None

    def __call__(self, *, prompt: str, negative_prompt: str | None, seed: int,
                 size: int, steps: int, guidance: float) -> Any:
        """返回一张 size×size 的 PIL 图像。"""


# ---------------------------------------------------------------- 不依赖模型的部分

def validate_name(name: Any) -> str:
    """名字要直接当文件名用，所以只收小写字母、数字、下划线。"""
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise TextureError(
            f"材质名 {name!r} 不合法：只能用小写字母、数字和下划线（^[a-z0-9_]+$），"
            "因为它会被直接用作文件名")
    return name


def build_prompt(prompt: str) -> str:
    """把用户写的那一小句套进模板，得到真正发给模型的整句。"""
    text = " ".join(str(prompt).split()).strip(" ,.")
    if not text:
        raise TextureError("prompt 是空的")
    return PROMPT_TEMPLATE.format(prompt=text)


def model_defaults(model: str) -> dict:
    """按模型给步数和引导系数的默认值。

    turbo 一类的蒸馏模型几步就够，而且训练时就没有无分类器引导，引导系数必须是 0，
    负面提示词在这种设置下根本不会被用到；普通的潜空间扩散模型则要二三十步加引导。
    两步是实测挑的：一步偏糊，四步开始过度锐化，素色墙面上会冒出亮斑。
    """
    if "turbo" in model.lower():
        return {"steps": 2, "guidance": 0.0}
    return {"steps": 25, "guidance": 7.0}


def validate_materials(materials: Any) -> dict[str, dict]:
    """一次查完所有条目，有错就在动手生成之前全报出来。"""
    if not isinstance(materials, dict):
        raise TextureError("materials 必须是 名字 -> {prompt, seed} 的字典")
    clean: dict[str, dict] = {}
    for name, entry in materials.items():
        validate_name(name)
        if not isinstance(entry, dict):
            raise TextureError(f"材质 {name}：条目必须是字典，拿到的是 {type(entry).__name__}")
        prompt, seed = entry.get("prompt"), entry.get("seed")
        if not isinstance(prompt, str) or not prompt.strip():
            raise TextureError(f"材质 {name}：prompt 必须是非空字符串")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2 ** 32:
            raise TextureError(f"材质 {name}：seed 必须是 0 到 2^32-1 之间的整数，拿到的是 {seed!r}")
        clean[name] = {"prompt": prompt, "seed": seed}
    return clean


def seam_error(image: Any) -> dict:
    """量一张图首尾相接时接缝有多明显。纯 numpy。

    把「最后一列对第一列」的平均绝对差（接缝），和图内相邻两列的平均绝对差（内部）放在一起比，
    行方向同理。能平铺的图，接缝差和内部差是同一个量级，比值在 1 上下；
    不能平铺的图，比值会到几倍、十几倍。

    光有这一条不够：纹理很碎的图，随便两列的差也大不到哪去，比值天然接近 1。
    所以再报两条标尺：

    * `interior_*_max`：图内最「跳」的那一对相邻列（行）。接缝不该比图里本来就有的任何一条边更显眼。
    * `unrelated_*`：相隔很远（八分之一到八分之七张图）的两列（行）的平均绝对差，
      也就是「两列毫不相干」时这个数是多少。
      接缝差贴着内部差、明显低于它，才说明首尾是真的接上了；接缝差贴着它，说明只是碰巧像。
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim != 3 or a.shape[0] < 2 or a.shape[1] < 2:
        raise ValueError(f"需要至少 2×2 的图，拿到的形状是 {a.shape}")
    out: dict[str, float] = {}
    for axis, tag in ((1, "x"), (0, "y")):
        lines = np.moveaxis(a, axis, 0)              # 第 0 维变成「列」（或「行」）
        step = np.abs(np.diff(lines, axis=0)).mean(axis=tuple(range(1, lines.ndim)))
        seam = float(np.abs(lines[0] - lines[-1]).mean())
        interior = float(step.mean())
        out[f"seam_{tag}"] = round(seam, 4)
        out[f"interior_{tag}"] = round(interior, 4)
        out[f"interior_{tag}_max"] = round(float(step.max()), 4)
        # 取七个错位（1/8 到 7/8 张图）求平均，免得规则图案恰好在某个错位上和自己重合
        n = len(lines)
        shifts = sorted({round(n * k / 8) % n for k in range(1, 8)} - {0})
        far = float(np.mean([np.abs(lines - np.roll(lines, k, axis=0)).mean() for k in shifts]))
        out[f"unrelated_{tag}"] = round(far, 4)
        # 纯色图内部差为零：接缝也为零就算 1（无缝），否则给一个很大的数
        if interior > 1e-9:
            out[f"ratio_{tag}"] = round(seam / interior, 3)
        else:
            out[f"ratio_{tag}"] = 1.0 if seam <= 1e-9 else 1e9
    return out


def tile(image: Any, repeat: int = 2) -> np.ndarray:
    """把图横竖各重复 repeat 次，拼成预览。看接缝最直接的办法。"""
    a = np.asarray(image)
    reps = (repeat, repeat) + (1,) * (a.ndim - 2)
    return np.tile(a, reps)


def load_log(out_dir: Path) -> dict[str, dict]:
    """读上一次的 textures.json，按名字索引。读不出来就当没有，全部重画。"""
    path = Path(out_dir) / LOG_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return {r["name"]: r for r in data.get("textures", []) if isinstance(r, dict) and "name" in r}
    except (ValueError, AttributeError, TypeError):
        return {}


def is_current(record: dict | None, wanted: dict, png: Path) -> bool:
    """PNG 还在，而且日志里记的出图条件和这次要的一模一样，才算不用重画。"""
    if record is None or not png.is_file():
        return False
    return all(record.get(k) == wanted[k] for k in IDENTITY_KEYS)


def _write_log(out_dir: Path, model: str, records: list[dict]) -> None:
    """先写临时文件再改名，中途被打断也不会留下半截 JSON。"""
    body = {"model": model, "prompt_template": PROMPT_TEMPLATE, "textures": records}
    path = out_dir / LOG_NAME
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------- 模型相关

def make_circular(module: Any) -> int:
    """把模块里所有 Conv2d 换成环绕补边，返回改了几个。

    `Conv2d` 构造时不管什么补边方式都会算好 `_reversed_padding_repeated_twice`，
    `_conv_forward` 每次调用都现查 `padding_mode`，所以构造完再改这个属性就够了，
    不需要另外打补丁。这一点依赖 torch 的内部实现，所以不靠假设：
    `tests/test_bench_texture.py` 里用「先平移再卷积 = 先卷积再平移」直接验证。
    1×1 的卷积没有补边，改不改都一样，一并改掉省得分情况。
    """
    import torch

    changed = 0
    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            if isinstance(layer.padding, str):
                raise RuntimeError(f"不认识的补边写法 padding={layer.padding!r}，环绕补边不保证正确")
            layer.padding_mode = "circular"
            # F.pad 真正用的补边量。按当前 padding 重算一遍，不指望构造时留下的值
            layer._reversed_padding_repeated_twice = tuple(
                p for pad in reversed(layer.padding) for p in (pad, pad))
            changed += 1
    return changed


def pick_device(device: str | None = None) -> str:
    """没指定就按 MPS、CUDA、CPU 的顺序挑。"""
    if device:
        return device
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_model(model: str, models_dir: Path = MODELS_DIR) -> tuple[Path, str | None, str | None]:
    """把模型名变成本地目录，返回 (目录, 版本哈希, 权重变体)。

    先只看本地缓存，缓存里没有才联网下载；下载是匿名的（`token=False`），
    需要登录或者点同意许可才能下的模型在这里会直接失败，这是有意的。
    优先要 fp16 的 safetensors，下载量小一半；模型没有这个变体就退回完整权重。
    """
    local = Path(model)
    if local.is_dir():
        return local, None, ("fp16" if any(local.rglob("*.fp16.safetensors")) else None)

    from diffusers import DiffusionPipeline

    models_dir.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for local_only in (True, False):
        for variant in ("fp16", None):
            try:
                # safety_checker=None：材质贴图用不上内容过滤器，不下那六百兆
                folder = Path(DiffusionPipeline.download(
                    model, variant=variant, use_safetensors=True, cache_dir=str(models_dir),
                    token=False, local_files_only=local_only, safety_checker=None))
            except Exception as error:  # 缓存里没有、没有这个变体、网络不通，都换下一种试
                last = error
                continue
            # 缓存布局是 models--组织--名字/snapshots/<版本哈希>/
            revision = folder.name if folder.parent.name == "snapshots" else None
            return folder, revision, variant
    raise RuntimeError(f"模型 {model} 拿不到（匿名下载失败，本地缓存里也没有）：{last}")


class DiffusersRenderer:
    """本机的扩散模型。加载一次，后面每张图复用。"""

    def __init__(self, model: str, device: str | None = None, models_dir: Path = MODELS_DIR):
        import torch
        from diffusers import AutoPipelineForText2Image

        self.device = pick_device(device)
        folder, self.revision, variant = resolve_model(model, models_dir)
        # 半精度只在加速设备上用；CPU 上半精度又慢又容易出错
        dtype = torch.float16 if self.device in ("mps", "cuda") else torch.float32
        index = json.loads((folder / "model_index.json").read_text(encoding="utf-8"))
        extra = ({"safety_checker": None, "requires_safety_checker": False}
                 if "safety_checker" in index else {})
        with warnings.catch_warnings():
            # diffusers 新版本把 torch_dtype 改名叫 dtype，老版本只认旧名字；两边都要能跑
            warnings.simplefilter("ignore", FutureWarning)
            pipe = AutoPipelineForText2Image.from_pretrained(
                str(folder), torch_dtype=dtype, variant=variant, use_safetensors=True,
                local_files_only=True, **extra)
        self.circular_layers = make_circular(pipe.unet) + make_circular(pipe.vae)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe.to(self.device)
        self._torch = torch

    def __call__(self, *, prompt: str, negative_prompt: str | None, seed: int,
                 size: int, steps: int, guidance: float) -> Any:
        # 生成器放在 CPU 上：初始噪声和每一步加的噪声都从这条流里来，再搬到设备上。
        generator = self._torch.Generator(device="cpu").manual_seed(int(seed))
        extra = {"negative_prompt": negative_prompt} if negative_prompt else {}
        with self._torch.inference_mode():
            result = self.pipe(prompt=prompt, height=size, width=size, num_inference_steps=steps,
                               guidance_scale=guidance, generator=generator, **extra)
        return result.images[0].convert("RGB")


# ---------------------------------------------------------------- 主流程

def generate_textures(materials: dict, out_dir: Path, *, model: str, size: int = 512,
                      steps: int | None = None, device: str | None = None,
                      renderer_factory: Callable[[], Renderer] | None = None) -> dict:
    """给 materials 里每一条生成 `out_dir/<名字>.png`，并写 `out_dir/textures.json`。

    `materials` 是 `名字 -> {"prompt": str, "seed": int, ...}`，多余的键不看。
    返回的日志里 `textures` 是每张图的完整记录（和 textures.json 里的一样），
    `generated` / `skipped` 是这一次实际画了哪些、跳过了哪些。

    `renderer_factory` 留给测试：给一个不带参数、返回出图函数的可调用对象，就不会去加载模型。
    只有确实有图要画的时候才会调用它，所以全部命中缓存的重跑是瞬时的。
    """
    entries = validate_materials(materials)
    if size <= 0 or size % 64:
        # 网络里有三次 2 倍下采样，潜空间边长在每一层都得是偶数，环绕补边才严格成立
        raise TextureError(f"size 必须是 64 的正整数倍，拿到的是 {size}")
    defaults = model_defaults(model)
    steps = int(steps) if steps is not None else defaults["steps"]
    if steps <= 0:
        raise TextureError(f"steps 必须是正整数，拿到的是 {steps}")
    guidance = float(defaults["guidance"])
    negative = NEGATIVE_PROMPT if guidance > 1.0 else None

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    previous = load_log(out_dir)
    started = time.time()
    renderer: Renderer | None = None
    records: dict[str, dict] = {}
    generated: list[str] = []
    skipped: list[str] = []

    # 先把能沿用的记录都放进去，这样中途写出的日志始终是完整的
    wanted: dict[str, dict] = {}
    for name, entry in entries.items():
        wanted[name] = {"full_prompt": build_prompt(entry["prompt"]), "negative_prompt": negative,
                        "seed": entry["seed"], "model": model, "size": size,
                        "steps": steps, "guidance": guidance}
        if is_current(previous.get(name), wanted[name], out_dir / f"{name}.png"):
            records[name] = previous[name]
            skipped.append(name)

    for name, entry in entries.items():
        if name in records:
            continue
        if renderer is None:
            renderer = renderer_factory() if renderer_factory else DiffusersRenderer(model, device)
        want = wanted[name]
        t0 = time.time()
        image = renderer(prompt=want["full_prompt"], negative_prompt=negative, seed=want["seed"],
                         size=size, steps=steps, guidance=guidance)
        image = image.convert("RGB")
        if image.size != (size, size):
            raise RuntimeError(f"材质 {name}：模型给的图是 {image.size}，要的是 {(size, size)}")
        pixels = np.asarray(image)
        if int(pixels.max()) - int(pixels.min()) == 0:
            # 半精度在某些设备上会算出 NaN，解码出来就是一张纯色图。宁可停下也不留一张坏图。
            raise RuntimeError(f"材质 {name}：生成结果是纯色图，多半是数值溢出，请换设备或换模型")
        image.save(out_dir / f"{name}.png")
        records[name] = {
            "name": name, "file": f"{name}.png", "prompt": entry["prompt"], **want,
            "revision": getattr(renderer, "revision", None),
            "device": getattr(renderer, "device", None),
            "seconds": round(time.time() - t0, 3),
            "seam": seam_error(pixels),
        }
        generated.append(name)
        _write_log(out_dir, model, [records[n] for n in entries if n in records])

    ordered = [records[n] for n in entries]
    _write_log(out_dir, model, ordered)
    return {"model": model, "out_dir": str(out_dir), "log": str(out_dir / LOG_NAME),
            "textures": ordered, "generated": generated, "skipped": skipped,
            "seconds": round(time.time() - started, 3)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m gwm.bench.texture",
        description="按 materials.json 在本机生成可平铺的贴图。")
    parser.add_argument("--materials", required=True, type=Path, help="materials.json 的路径")
    parser.add_argument("--out", required=True, type=Path, help="输出目录")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"模型名或本地目录（默认 {DEFAULT_MODEL}）")
    parser.add_argument("--size", type=int, default=512, help="边长，64 的倍数（默认 512）")
    parser.add_argument("--steps", type=int, default=None, help="采样步数（默认按模型定）")
    parser.add_argument("--device", default=None, help="mps / cuda / cpu（默认自动挑）")
    args = parser.parse_args(argv)

    try:
        materials = json.loads(args.materials.read_text(encoding="utf-8"))
        log = generate_textures(materials, args.out, model=args.model, size=args.size,
                                steps=args.steps, device=args.device)
    except (TextureError, OSError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2
    for record in log["textures"]:
        state = "跳过" if record["name"] in log["skipped"] else f"{record['seconds']:.1f}s"
        seam = record.get("seam", {})
        print(f"{record['name']:<24} {state:>8}  接缝/内部 x={seam.get('ratio_x')} y={seam.get('ratio_y')}")
    print(f"生成 {len(log['generated'])} 张，跳过 {len(log['skipped'])} 张，"
          f"共 {log['seconds']:.1f}s -> {log['log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
