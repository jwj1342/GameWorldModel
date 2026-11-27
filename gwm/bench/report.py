"""把若干次 eval 的结果拼成一个静态网页。

整页只想说清楚一张二维图：横轴是「能不能动」（交互清单通过率），纵轴是「像不像」
（留出视角的 LPIPS）。两根轴各管各的，任何地方都不把它们合成一个分数。

几条规矩：

- 产物是 `out_dir/index.html` 加一个 `out_dir/img/`，全部内联，不引用任何外部脚本、
  字体或样式，用 file:// 直接打开就能看。
- 图是 Python 里拼出来的内联 SVG，不用画图库。颜色只表示渲染模式（纯色、贴图两种），
  哪个点是哪次运行靠直接写在点旁边的名字，所以运行再多也不会用到第三种颜色。
- 引用点（配置里的 `references`）不是这台机器上算出来的数，图上画成空心、
  表里单独标「引用」，并带上来源链接。
- 所有写进页面的字符串（运行编号、提示词、备注）都当作不可信输入，一律转义。
- 缺什么就说缺什么：没有 LPIPS 的点不画进图里，但会在图下面点名，表里写「未计算」。
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import math
import re
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[2]

GALLERY_VIEWS = 6           # 画廊里抽几个留出视角
IMAGE_WIDTH = 640           # 页面用图的宽度上限（像素）
SWATCH_WIDTH = 256          # 贴图样片的宽度上限
JPEG_QUALITY = 85

MODE_NAMES = {"flat": "纯色", "textured": "贴图"}

EXPECT_LEGEND = [
    ("movable", "抓住后提起横拖或直接平推，四个水平方向里有一种跟得住手、松手后落稳"),
    ("fixed", "推不动"),
    ("rests", "放开 2.5 秒后位移不超过 5 厘米"),
    ("hinge", "绕轴转且卡得住限位"),
    ("slide", "沿轴滑且卡得住限位"),
]

# 原因码的含义，和 harness/bench_interact.mjs 里的判定一一对应。
REASON_LEGEND = {
    "missing_object": "场景里没有这个名字的对象",
    "declared_static": "对象在，但被声明成抓不动的静态体",
    "declared_movable": "清单要求固定，场景里却声明成可以抓取",
    "drag_failed": "抓取、提起、横拖、松手这一套执行下来没有通过",
    "constraint_failed": "约束类型声明对了，但执行时没有沿约束运动，或者没卡住限位",
    "moved_after_release": "放开后位移超过 5 厘米",
    "tipped_over": "放开后倾倒，转角超过 10 度",
    "fell_through": "放开后穿过地面掉了下去",
    "not_finite": "仿真结果里出现了非有限数",
}


# ---------------------------------------------------------------- 小工具

def _esc(value) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _read_json(path: Path, default=None):
    """读不到或者读坏了都返回 default：汇总页不该因为某次运行缺一个文件就整页出不来。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _number(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _fmt(value, digits: int) -> str:
    number = _number(value)
    return "未计算" if number is None else f"{number:.{digits}f}"


def _percent(rate) -> str:
    number = _number(rate)
    return "—" if number is None else f"{number * 100:.0f}%"


def _safe_stem(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(text))[:60] or "x"


def _write_jpeg(src: Path, dst: Path, max_width: int) -> tuple[int, int] | None:
    """缩到 max_width 宽存成 JPEG，返回缩后的尺寸；源图读不了就返回 None。"""
    try:
        with Image.open(src) as image:
            image = image.convert("RGB")
            if image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.LANCZOS)
            dst.parent.mkdir(parents=True, exist_ok=True)
            image.save(dst, "JPEG", quality=JPEG_QUALITY, optimize=True)
            return image.size
    except (OSError, ValueError):
        return None


def _link(url, text=None) -> str:
    """只有 http(s) 的地址才做成链接，别的协议（javascript: 之类）原样当文字显示。"""
    url = "" if url is None else str(url).strip()
    if not url:
        return ""
    label = _esc(text if text is not None else url)
    if re.match(r"https?://", url, re.IGNORECASE):
        return f'<a href="{_esc(url)}" rel="noopener noreferrer">{label}</a>'
    return label


# ---------------------------------------------------------------- 读数据

def _load_run(run_dir: Path, index: int) -> dict:
    run_dir = Path(run_dir)
    results = _read_json(run_dir / "results.json", {}) or {}
    meta = _read_json(run_dir / "run.json", {}) or {}
    interact = _read_json(run_dir / "interact.json", {}) or {}
    label = str(results.get("label") or meta.get("label") or run_dir.name)
    run = {
        "index": index, "dir": run_dir, "run_id": str(results.get("run_id") or meta.get("run_id") or run_dir.name),
        "label": label, "results": results, "meta": meta, "interact": interact,
        "fidelity": {}, "render_files": {}, "points": [],
    }
    code = results.get("code_size") if isinstance(results.get("code_size"), dict) else {}
    for raw in results.get("points") or []:
        if not isinstance(raw, dict):
            continue
        mode = str(raw.get("mode") or "flat")
        fidelity = raw.get("fidelity") if isinstance(raw.get("fidelity"), dict) else {}
        summary = raw.get("interact") if isinstance(raw.get("interact"), dict) else (interact.get("summary") or {})
        passed, total = _number(summary.get("passed")), _number(summary.get("total"))
        rate = passed / total if passed is not None and total else _number(summary.get("pass_rate"))
        per_view = _read_json(run_dir / f"fidelity_{mode}.json", {}) or {}
        run["fidelity"][mode] = {str(r.get("name")): r for r in per_view.get("per_view") or [] if isinstance(r, dict)}
        frames = (_read_json(run_dir / "renders" / mode / "index.json", {}) or {}).get("frames") or []
        run["render_files"][mode] = {str(f.get("name")): str(f.get("file")) for f in frames if isinstance(f, dict)}
        run["points"].append({
            "kind": "measured", "run": run, "mode": mode,
            "name": str(raw.get("name") or f"{label}-{mode}"),
            "psnr": _number(fidelity.get("psnr")), "ssim": _number(fidelity.get("ssim")),
            "lpips": _number(fidelity.get("lpips")),
            "passed": None if passed is None else int(passed), "total": None if total is None else int(total),
            "pass_rate": rate, "lines": code.get("lines"), "materials": results.get("materials"),
            "texture_model": results.get("texture_model") if mode == "textured" else None,
        })
    return run


def _load_references(cfg: dict) -> list[dict]:
    points = []
    for raw in cfg.get("references") or []:
        if not isinstance(raw, dict):
            continue
        fidelity = raw.get("fidelity") if isinstance(raw.get("fidelity"), dict) else {}
        points.append({
            "kind": "cited", "run": None, "mode": None, "name": str(raw.get("name") or "引用点"),
            "psnr": _number(fidelity.get("psnr")), "ssim": _number(fidelity.get("ssim")),
            "lpips": _number(fidelity.get("lpips")), "passed": None, "total": None,
            "pass_rate": _number(raw.get("interact_pass_rate")), "lines": None, "materials": None,
            "texture_model": None, "note": raw.get("note"), "source": raw.get("source"),
        })
    return points


def _dedupe_names(points: list[dict]) -> None:
    """两次运行用了同一个 label 时点名会撞，撞了就把运行编号补在后面。"""
    seen: dict[str, int] = {}
    for point in points:
        seen[point["name"]] = seen.get(point["name"], 0) + 1
    for point in points:
        if seen[point["name"]] > 1 and point["run"] is not None:
            point["name"] = f"{point['name']} · {point['run']['run_id']}"


def _pick_views(views: list[dict], count: int = GALLERY_VIEWS) -> list[dict]:
    """按顺序均匀抽 count 个，首尾都取。"""
    n = len(views)
    if n <= count:
        return list(views)
    return [views[i] for i in sorted({round(k * (n - 1) / (count - 1)) for k in range(count)})]


# ---------------------------------------------------------------- 二维图

def _text_width(text: str, size: float) -> float:
    """估文字宽度：中日韩字符按一个字宽，大写字母和数字按 0.64，其余按 0.52。只用来摆标签，不求精确。"""
    total = 0.0
    for ch in text:
        total += size if ord(ch) > 0x2E7F else size * (0.64 if ch.isupper() or ch.isdigit() else 0.52)
    return total


def _overlap(a: tuple, b: tuple) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def _label_candidates(x: float, y: float, w: float, h: float):
    """一个点的标签可以放哪。先试紧挨着的八个方位，放不下再往外拉开，拉开的要画引线。"""
    g = 11.0
    yield (x + g, y - h / 2, x + g + w, y + h / 2), False
    yield (x - g - w, y - h / 2, x - g, y + h / 2), False
    yield (x - w / 2, y - g - h, x + w / 2, y - g), False
    yield (x - w / 2, y + g, x + w / 2, y + g + h), False
    d = g * 0.75
    yield (x + d, y - d - h, x + d + w, y - d), False
    yield (x + d, y + d, x + d + w, y + d + h), False
    yield (x - d - w, y - d - h, x - d, y - d), False
    yield (x - d - w, y + d, x - d, y + d + h), False
    for radius in (28.0, 44.0, 62.0, 84.0, 110.0):
        for degrees in (-35, 35, 145, 215, -65, 65, 115, 245, -90, 90):
            ax = x + radius * math.cos(math.radians(degrees))
            ay = y + radius * math.sin(math.radians(degrees))
            cos = math.cos(math.radians(degrees))
            x0 = ax if cos > 0.3 else ax - w if cos < -0.3 else ax - w / 2
            yield (x0, ay - h / 2, x0 + w, ay + h / 2), True


def _gap(box: tuple, x: float, y: float) -> float:
    """点到矩形的距离，点在矩形里面时是 0。"""
    return math.hypot(max(box[0] - x, 0.0, x - box[2]), max(box[1] - y, 0.0, y - box[3]))


def _place_labels(marks: list[dict], bounds: tuple, obstacles: list[tuple], size: float) -> None:
    """贪心地给每个点找一个不压到别的点、别的标签、也不出界的位置，结果写回 marks。

    几个点挤在一起时，光是「不重叠」还不够：标签要是离别人的点比离自己的点还近，读的人会认错。
    所以这种位置要重罚，宁可拉远了画一根引线。
    """
    blocked = list(obstacles) + [(m["x"] - 9, m["y"] - 9, m["x"] + 9, m["y"] + 9) for m in marks]
    h = size + 4
    for mark in marks:
        w = _text_width(mark["label"], size) + 4
        own = (mark["x"] - 9, mark["y"] - 9, mark["x"] + 9, mark["y"] + 9)
        others = [m for m in marks if m is not mark]
        best = None
        for order, (box, leader) in enumerate(_label_candidates(mark["x"], mark["y"], w, h)):
            outside = (max(0.0, bounds[0] - box[0]) + max(0.0, box[2] - bounds[2])
                       + max(0.0, bounds[1] - box[1]) + max(0.0, box[3] - bounds[3]))
            cost = outside * 1000 + sum(_overlap(box, other) for other in blocked if other != own) + order * 0.01
            mine = _gap(box, mark["x"], mark["y"])
            if any(_gap(box, m["x"], m["y"]) < mine + 3 for m in others):
                cost += 400
            if best is None or cost < best[0]:
                best = (cost, box, leader)
            if cost < 1.0:
                break
        _, box, leader = best
        mark["box"], mark["leader"] = box, leader
        blocked.append(box)


def _nice_lpips_axis(values: list[float]) -> tuple[float, float]:
    """纵轴从 0 起，上限取到比最大值多一点的整刻度；返回 (上限, 刻度间隔)。"""
    top = max(values) if values else 0.5
    step = 0.2 if top + 0.04 > 0.6 else 0.1
    upper = max(0.5, math.ceil((top + 0.04) / step - 1e-9) * step)
    return round(upper, 6), step


def _marker(point: dict, x: float, y: float) -> str:
    if point["kind"] == "cited":
        return f'<circle class="mk-cited" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'
    if point["mode"] == "textured":
        r = 7.5
        path = f"M{x:.1f} {y - r:.1f}L{x + r:.1f} {y:.1f}L{x:.1f} {y + r:.1f}L{x - r:.1f} {y:.1f}Z"
        return f'<path class="mk-textured" d="{path}"/>'
    return f'<circle class="mk-flat" cx="{x:.1f}" cy="{y:.1f}" r="6"/>'


def _tooltip_text(point: dict) -> str:
    lines = [point["name"] + ("（引用，非本机测得）" if point["kind"] == "cited" else "")]
    if point["kind"] == "measured":
        lines.append(f"模式：{MODE_NAMES.get(point['mode'], point['mode'])}（{point['mode']}）")
    lines.append(f"LPIPS {_fmt(point['lpips'], 3)}　PSNR {_fmt(point['psnr'], 2)}　SSIM {_fmt(point['ssim'], 3)}")
    if point["passed"] is not None and point["total"] is not None:
        lines.append(f"交互清单 {point['passed']}/{point['total']}（{_percent(point['pass_rate'])}）")
    else:
        lines.append(f"交互清单通过率 {_percent(point['pass_rate'])}")
    return "\n".join(lines)


def _chart_svg(points: list[dict], width: int, height: int, css_class: str) -> str:
    """画一张图。points 里只有两根轴都有数的点。纵轴朝上是 LPIPS 变小，也就是更像。"""
    size = 12.0
    left, right, top, bottom = (46, 24, 54, 50) if width >= 600 else (36, 24, 54, 50)
    x0, x1, y0, y1 = left, width - right, top, height - bottom
    upper, step = _nice_lpips_axis([p["lpips"] for p in points])

    def sx(rate: float) -> float:
        return x0 + min(max(rate, 0.0), 1.0) * (x1 - x0)

    def sy(lpips: float) -> float:
        return y0 + min(max(lpips, 0.0), upper) / upper * (y1 - y0)

    out = [f'<svg class="chart {css_class}" viewBox="0 0 {width} {height}" role="img" '
           f'aria-label="二维图：横轴交互清单通过率，纵轴 LPIPS（越靠上越低、越像照片）。具体数值见下方汇总表。">']
    # 网格和刻度：实线细线，颜色只比底色差一档
    ticks_y = [round(i * step, 6) for i in range(int(round(upper / step)) + 1)]
    for value in ticks_y:
        y = sy(value)
        out.append(f'<line class="grid" x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick" x="{x0 - 8}" y="{y + 4:.1f}" text-anchor="end">{value:.1f}</text>')
    for rate in (0, 0.25, 0.5, 0.75, 1.0):
        x = sx(rate)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{y0}" x2="{x:.1f}" y2="{y1}"/>')
        out.append(f'<text class="tick" x="{x:.1f}" y="{y1 + 18}" text-anchor="middle">{rate * 100:.0f}%</text>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}"/>')
    out.append(f'<line class="axis" x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}"/>')
    # 轴标题：两根轴都把「哪边更好」直接写出来
    out.append(f'<text class="axis-title" x="{x0 - (left - 4)}" y="16">LPIPS（越低越像）</text>')
    out.append(f'<text class="axis-note" x="{x0 - (left - 4)}" y="32">↑ 越靠上越像照片</text>')
    out.append(f'<text class="axis-title" x="{x1}" y="{height - 8}" text-anchor="end">交互清单通过率（越靠右越能动）→</text>')
    corner = "又像又能动 ↗"
    corner_w = _text_width(corner, size)
    out.append(f'<text class="axis-note" x="{x1 - 6}" y="{y0 + 16}" text-anchor="end">{corner}</text>')

    marks = [{"point": p, "x": sx(p["pass_rate"]), "y": sy(p["lpips"]),
              "label": p["name"] + ("（引用）" if p["kind"] == "cited" else "")} for p in points]
    # 同一次运行的两种模式共用一份场景代码，交互分数必然相同：用一根细线把它们连起来
    obstacles = [(x1 - 8 - corner_w, y0 + 2, x1, y0 + 22)]
    by_run: dict[int, list[dict]] = {}
    for mark in marks:
        if mark["point"]["run"] is not None:
            by_run.setdefault(mark["point"]["run"]["index"], []).append(mark)
    for group in by_run.values():
        if len(group) > 1:
            ys = [m["y"] for m in group]
            x = group[0]["x"]
            out.append(f'<line class="link" x1="{x:.1f}" y1="{min(ys):.1f}" x2="{x:.1f}" y2="{max(ys):.1f}"/>')
            obstacles.append((x - 3, min(ys), x + 3, max(ys)))
    _place_labels(marks, (x0 + 3, y0 + 2, width - 3, y1 - 3), obstacles, size)

    for mark in marks:
        box = mark["box"]
        if mark["leader"]:
            tx = min(max(mark["x"], box[0]), box[2])
            ty = min(max(mark["y"], box[1]), box[3])
            out.append(f'<line class="leader" x1="{mark["x"]:.1f}" y1="{mark["y"]:.1f}" x2="{tx:.1f}" y2="{ty:.1f}"/>')
    for mark in marks:
        out.append(_marker(mark["point"], mark["x"], mark["y"]))
    for mark in marks:
        box = mark["box"]
        out.append(f'<text class="pt-label" x="{box[0] + 2:.1f}" y="{box[3] - 5:.1f}">{_esc(mark["label"])}</text>')
    # 悬停和键盘聚焦的热区比点本身大，数值同时也在汇总表里，不靠悬停才看得到
    for mark in marks:
        tip = _tooltip_text(mark["point"])
        out.append(f'<circle class="hit" cx="{mark["x"]:.1f}" cy="{mark["y"]:.1f}" r="14" tabindex="0" '
                   f'role="img" aria-label="{_esc(tip)}" data-tip="{_esc(tip)}"/>')
    out.append("</svg>")
    return "".join(out)


def _chart_section(points: list[dict]) -> str:
    plotted = [p for p in points if p["lpips"] is not None and p["pass_rate"] is not None]
    skipped = [p for p in points if p not in plotted]
    parts = ['<section id="chart"><h2>两轴位置</h2>',
             '<p class="lead">每个点是一次运行的一种渲染模式。越靠右，清单里执行通过的条目越多；'
             '越靠上，留出视角的渲染和真实照片在感知上越接近。两根轴不合成一个分数。</p>']
    if plotted:
        kinds = []
        if any(p["kind"] == "measured" and p["mode"] != "textured" for p in plotted):
            kinds.append('<span class="key"><svg viewBox="0 0 16 16" aria-hidden="true"><circle class="mk-flat" cx="8" cy="8" r="6"/></svg>纯色渲染（flat）</span>')
        if any(p["kind"] == "measured" and p["mode"] == "textured" for p in plotted):
            kinds.append('<span class="key"><svg viewBox="0 0 16 16" aria-hidden="true"><path class="mk-textured" d="M8 0.5L15.5 8L8 15.5L0.5 8Z"/></svg>生成贴图（textured）</span>')
        if any(p["kind"] == "cited" for p in plotted):
            kinds.append('<span class="key"><svg viewBox="0 0 16 16" aria-hidden="true"><circle class="mk-cited" cx="8" cy="8" r="6"/></svg>引用点：数字来自外部文献，不是本机测得</span>')
        linked = len({p["run"]["index"] for p in plotted if p["run"] is not None}) < sum(1 for p in plotted if p["run"] is not None)
        if linked:
            kinds.append('<span class="key"><svg viewBox="0 0 16 16" aria-hidden="true"><line class="link" x1="8" y1="1" x2="8" y2="15"/></svg>竖线连着同一份场景代码的两种渲染</span>')
        parts.append('<div class="card chart-card">')
        if len(kinds) > 1:
            parts.append(f'<div class="legend">{"".join(kinds)}</div>')
        parts.append(_chart_svg(plotted, 760, 440, "chart-wide"))
        parts.append(_chart_svg(plotted, 360, 400, "chart-narrow"))
        parts.append('</div>')
        if any(p["kind"] == "measured" for p in plotted) and len(kinds) <= 1:
            only = plotted[0]
            what = "引用点（非本机测得）" if only["kind"] == "cited" else f"{MODE_NAMES.get(only['mode'], only['mode'])}渲染（{_esc(only['mode'])}）"
            parts.append(f'<p class="note">图上的点都是{what}。</p>')
    else:
        parts.append('<p class="card note">没有任何一个点同时带有 LPIPS 和交互通过率，这张图画不出来。数值见下面的汇总表。</p>')
    if skipped:
        names = "、".join(_esc(p["name"]) for p in skipped)
        parts.append(f'<p class="note">以下的点缺 LPIPS 或交互通过率（例如评测时跳过了 LPIPS），没有画进图里：{names}。</p>')
    parts.append('</section>')
    return "".join(parts)


# ---------------------------------------------------------------- 各个小节

def _header(cfg: dict, views: list[dict], image_dir: str) -> str:
    resolution = ""
    if views and _number(views[0].get("width")) and _number(views[0].get("height")):
        resolution = f"，{int(views[0]['width'])}×{int(views[0]['height'])} 像素"
    return (
        f'<header><h1>两轴评测：{_esc(cfg.get("scene", ""))}</h1>'
        '<p class="lead">同一个场景分两根轴看：<b>保真度</b>是在留出视角渲染后和真实照片比'
        '（PSNR、SSIM、LPIPS），<b>可交互性</b>是把一份人工写的物理事实清单放进物理引擎逐条执行，看通过率。</p>'
        f'<p class="meta">留出视角 {len(views)} 个 · 评测分辨率 {_esc(image_dir)}{resolution} · '
        f'生成日期 {_dt.date.today().isoformat()}</p></header>')


def _summary_table(points: list[dict]) -> str:
    rows = []
    for p in points:
        cited = p["kind"] == "cited"
        badge = '<span class="badge">引用</span>' if cited else ""
        mode = "—" if cited else f"{MODE_NAMES.get(p['mode'], p['mode'])}（{_esc(p['mode'])}）"
        if p["passed"] is not None and p["total"] is not None:
            checklist = f"{p['passed']}/{p['total']}（{_percent(p['pass_rate'])}）"
        else:
            checklist = _percent(p["pass_rate"])
        if cited:
            source = _link(p.get("source"), "来源")
            note = _esc(p.get("note") or "")
            remark = "非本机测得。" + note + (f" {source}" if source else "")
        else:
            remark = _esc(p["run"]["run_id"])
        rows.append(
            f'<tr{" class=cited" if cited else ""}><th scope="row">{_esc(p["name"])}{badge}</th><td>{mode}</td>'
            f'<td class="num">{_fmt(p["psnr"], 2)}</td><td class="num">{_fmt(p["ssim"], 3)}</td>'
            f'<td class="num">{_fmt(p["lpips"], 3)}</td><td class="num">{checklist}</td>'
            f'<td class="num">{_esc(p["lines"]) if p["lines"] is not None else "—"}</td>'
            f'<td class="num">{_esc(p["materials"]) if p["materials"] is not None else "—"}</td>'
            f'<td>{_esc(p["texture_model"]) if p["texture_model"] else "—"}</td><td class="remark">{remark}</td></tr>')
    return (
        '<section id="summary"><h2>汇总表</h2><div class="scroll card"><table><thead><tr>'
        '<th scope="col">点</th><th scope="col">模式</th><th scope="col" class="num">PSNR ↑</th>'
        '<th scope="col" class="num">SSIM ↑</th><th scope="col" class="num">LPIPS ↓</th>'
        '<th scope="col" class="num">清单通过</th><th scope="col" class="num">代码行数</th>'
        '<th scope="col" class="num">材质数</th><th scope="col">贴图模型</th><th scope="col">运行 / 备注</th>'
        f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'
        '<p class="note">↑ 越高越好，↓ 越低越好。主看 LPIPS：生成的贴图是「合理」而不是「相同」，PSNR 和 SSIM 天然吃亏，'
        '一起报是为了能和别处的表对上。代码行数不含空行和整行注释。</p></section>')


def _gallery(pack_dir: Path, views: list[dict], points: list[dict], img_dir: Path) -> str:
    measured = [p for p in points if p["kind"] == "measured"]
    chosen = _pick_views(views)
    if not chosen:
        return ""
    rows = []
    for view in chosen:
        name = str(view.get("name") or "")
        stem = Path(name).stem
        figures = []
        gt_src = pack_dir / "eval" / str(view.get("file") or f"gt/{stem}.png")
        gt_name = f"gt_{_safe_stem(stem)}.jpg"
        figures.append(_figure(gt_src, img_dir / gt_name, "真实照片（留出）", "", f"{name} 的真实照片"))
        for p in measured:
            run, mode = p["run"], p["mode"]
            file = run["render_files"].get(mode, {}).get(name) or f"{stem}.png"
            dst = f"r{run['index']}_{_safe_stem(mode)}_{_safe_stem(stem)}.jpg"
            row = run["fidelity"].get(mode, {}).get(name) or {}
            if _number(row.get("lpips")) is not None:
                metric = f"LPIPS {_fmt(row.get('lpips'), 3)}"
            elif _number(row.get("psnr")) is not None:
                metric = f"LPIPS 未计算 · PSNR {_fmt(row.get('psnr'), 2)}"
            else:
                metric = "LPIPS 未计算"
            figures.append(_figure(run["dir"] / "renders" / mode / file, img_dir / dst, p["name"], metric,
                                   f"{p['name']} 在 {name} 视角的渲染"))
        rows.append(f'<div class="view"><h3>{_esc(name)}</h3><div class="scroll strip" style="--n:{len(figures)}">'
                    f'{"".join(figures)}</div></div>')
    return (
        f'<section id="gallery"><h2>留出视角对照</h2><p class="lead">从 {len(views)} 个留出视角里按顺序均匀抽 {len(chosen)} 个。'
        '每行第一张是真实照片，后面是各个点在同一机位的渲染，图下是这一个视角的 LPIPS（越低越像）。</p>'
        f'{"".join(rows)}</section>')


def _figure(src: Path, dst: Path, title: str, metric: str, alt: str) -> str:
    size = _write_jpeg(src, dst, IMAGE_WIDTH) if Path(src).is_file() else None
    if size is None:
        body = '<div class="missing">缺图</div>'
    else:
        body = (f'<img src="img/{_esc(dst.name)}" width="{size[0]}" height="{size[1]}" '
                f'alt="{_esc(alt)}" loading="lazy">')
    extra = f'<span class="metric">{_esc(metric)}</span>' if metric else ""
    return f'<figure>{body}<figcaption><span class="who">{_esc(title)}</span>{extra}</figcaption></figure>'


def _checklist_section(runs: list[dict], checklist: dict | None) -> str:
    """清单一行一条。行以清单文件为准；文件读不到就退回到各次运行里实际执行过的条目。"""
    keys: list[tuple[str, str]] = []
    source = (checklist or {}).get("items") or [i for run in runs for i in run["interact"].get("items") or []]
    for item in source:
        if isinstance(item, dict):
            key = (str(item.get("target")), str(item.get("expect")))
            if key not in keys:
                keys.append(key)
    if not keys:
        return ""
    lookup = []
    for run in runs:
        table = {}
        for item in run["interact"].get("items") or []:
            if isinstance(item, dict):
                table[(str(item.get("target")), str(item.get("expect")))] = item
        lookup.append(table)
    reasons: list[str] = []
    head = ['<th scope="col" class="num">#</th><th scope="col">对象</th><th scope="col">期望</th>']
    for run in runs:
        summary = run["interact"].get("summary") or {}
        count = f'{summary.get("passed", "?")}/{summary.get("total", "?")}'
        head.append(f'<th scope="col">{_esc(run["label"])}<span class="sub">{_esc(count)} 通过</span></th>')
    body = []
    for number, key in enumerate(keys, 1):
        cells = [f'<td class="num">{number}</td><th scope="row"><code>{_esc(key[0])}</code></th><td><code>{_esc(key[1])}</code></td>']
        for table in lookup:
            item = table.get(key)
            if item is None:
                cells.append('<td class="muted">未执行</td>')
            elif item.get("pass"):
                cells.append('<td><span class="ok" aria-hidden="true">✓</span> 通过</td>')
            else:
                reason = str(item.get("reason") or "unknown")
                if reason not in reasons:
                    reasons.append(reason)
                cells.append(f'<td><span class="bad" aria-hidden="true">✗</span> 未通过 <code>{_esc(reason)}</code></td>')
        body.append(f'<tr>{"".join(cells)}</tr>')

    caveat = ""
    if checklist:
        bits = [str(checklist.get(k)) for k in ("annotated_by", "note") if checklist.get(k)]
        if bits:
            caveat = f'<p class="caveat"><b>注意：</b>{_esc("。".join(b.rstrip("。") for b in bits))}。</p>'
    expects = "".join(f'<li><code>{name}</code>：{text}</li>' for name, text in EXPECT_LEGEND)
    reason_items = []
    for reason in reasons:
        text = REASON_LEGEND.get(reason)
        if text is None and reason.startswith("declared_"):
            text = f"声明的约束类型（{reason[len('declared_'):]}）和清单要求的不一致"
        reason_items.append(f'<li><code>{_esc(reason)}</code>：{_esc(text or "未收录的原因码，见 interact.json 里的 detail")}</li>')
    reason_block = f'<h3>出现过的原因码</h3><ul class="legend-list">{"".join(reason_items)}</ul>' if reason_items else ""
    return (
        '<section id="checklist"><h2>交互清单</h2>'
        '<p class="lead">清单不给写场景的智能体看。每条是现实里成立的一件事，评测时在物理引擎里逐条执行；'
        '同一次运行的纯色和贴图两种渲染共用这一列结果，因为交互只看几何。</p>'
        f'{caveat}<div class="scroll card"><table class="checks"><thead><tr>{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(body)}</tbody></table></div>'
        f'<h3>五种期望</h3><ul class="legend-list">{expects}</ul>{reason_block}</section>')


def _textures_section(runs: list[dict], img_dir: Path) -> str:
    blocks = []
    for run in runs:
        tex_dir = run["dir"] / "candidate" / "textures"
        if not tex_dir.is_dir():
            continue
        log = _read_json(tex_dir / "textures.json", {}) or {}
        records = [r for r in log.get("textures") or [] if isinstance(r, dict) and r.get("name")]
        if not records:
            # 没有日志就退回到目录里的图，提示词和种子从 materials.json 里找
            materials = _read_json(run["dir"] / "candidate" / "materials.json", {}) or {}
            for path in sorted(tex_dir.glob("*.png")):
                entry = materials.get(path.stem) if isinstance(materials.get(path.stem), dict) else {}
                records.append({"name": path.stem, "file": path.name, "prompt": entry.get("prompt"), "seed": entry.get("seed")})
        cards = []
        for record in records:
            name = str(record["name"])
            file = Path(str(record.get("file") or f"{name}.png")).name     # 只取文件名，不跟着日志里的路径走出目录
            dst = img_dir / f"tex_r{run['index']}_{_safe_stem(name)}.jpg"
            size = _write_jpeg(tex_dir / file, dst, SWATCH_WIDTH) if (tex_dir / file).is_file() else None
            image = (f'<img src="img/{_esc(dst.name)}" width="{size[0]}" height="{size[1]}" alt="材质 {_esc(name)} 的贴图" loading="lazy">'
                     if size else '<div class="missing">缺图</div>')
            seed = record.get("seed")
            cards.append(
                f'<figure class="swatch">{image}<figcaption><span class="who"><code>{_esc(name)}</code></span>'
                f'<span class="prompt">{_esc(record.get("prompt") or "（没有记录提示词）")}</span>'
                f'<span class="metric">种子 {_esc(seed) if seed is not None else "—"}</span></figcaption></figure>')
        if cards:
            model = log.get("model") or run["results"].get("texture_model")
            model_text = f' · 模型 <code>{_esc(model)}</code>' if model else ""
            blocks.append(f'<h3>{_esc(run["label"])}<span class="sub">{len(cards)} 张{model_text}</span></h3>'
                          f'<div class="swatches">{"".join(cards)}</div>')
    if not blocks:
        return ""
    return ('<section id="textures"><h2>生成的贴图</h2><p class="lead">场景代码里每种材质只写一句提示词和一个种子，'
            '贴图由本机的扩散模型按它们生成，可平铺。下面是实际用到的样片。</p>' + "".join(blocks) + '</section>')


def _agent_text(agent) -> str:
    """run.json 里的 agent 是 claude -p 的原始记录，字段不保证都在，有什么写什么。"""
    if not isinstance(agent, dict):
        return ""
    result = agent.get("result") if isinstance(agent.get("result"), dict) else {}
    model = agent.get("model") or result.get("model")
    command = agent.get("command")
    if not model and isinstance(command, list) and "--model" in command[:-1]:
        model = command[command.index("--model") + 1]
    cost = _number(result.get("total_cost_usd"))
    if cost is None:
        cost = _number(agent.get("total_cost_usd", agent.get("cost_usd", agent.get("cost"))))
    turns = _number(result.get("num_turns", agent.get("num_turns", agent.get("turns"))))
    seconds = _number(agent.get("seconds"))
    bits = []
    if model:
        bits.append(f"模型 {model}")
    if cost is not None:
        bits.append(f"花费 ${cost:.2f}")
    if turns is not None:
        bits.append(f"{int(turns)} 轮")
    if seconds is not None:
        bits.append(f"用时 {seconds / 60:.1f} 分钟")
    return "，".join(bits) if bits else "有记录，但没有可读的模型、花费或轮数字段"


def _provenance(runs: list[dict], frame: dict | None, checklist: dict | None, cfg: dict) -> str:
    items = []
    for run in runs:
        meta = run["meta"]
        agent = _agent_text(meta.get("agent"))
        agent_line = f'<dt>智能体</dt><dd>{_esc(agent)}</dd>' if agent else '<dt>智能体</dt><dd>无记录（场景不是由智能体运行产生的，或没有留下 agent_run.json）</dd>'
        items.append(
            f'<div class="run"><h3>{_esc(run["label"])}</h3><dl>'
            f'<dt>运行编号</dt><dd><code>{_esc(run["run_id"])}</code></dd>'
            f'<dt>git 提交</dt><dd><code>{_esc(meta.get("git_commit") or "未知")}</code></dd>'
            f'<dt>开始时间</dt><dd>{_esc(meta.get("started") or "未知")}</dd>{agent_line}</dl></div>')
    notes = []
    norm = (frame or {}).get("normalisation") if isinstance((frame or {}).get("normalisation"), dict) else None
    if norm:
        assumed = norm.get("camera_height_m_assumed")
        text = (f"坐标归一化：尺度是先验，不是量出来的。COLMAP 给不出绝对尺度，这里假设相机离地高度的中位数是 "
                f"{assumed} 米，由此得到缩放系数 {_fmt(norm.get('scale'), 4)}。")
        spread = norm.get("camera_height_spread_m")
        if isinstance(spread, list) and len(spread) == 3:
            text += f"按这个尺度，相机高度的 5% / 50% / 95% 分位是 {spread[0]} / {spread[1]} / {spread[2]} 米。"
        if _number(norm.get("up_vs_camera_up_deg")) is not None:
            text += f"拟合出的地面法线和相机平均朝上方向相差 {norm['up_vs_camera_up_deg']} 度。"
        notes.append(text)
    if frame and frame.get("split"):
        counts = frame.get("counts") if isinstance(frame.get("counts"), dict) else {}
        extra = f"（共 {counts.get('all')} 张，留出 {counts.get('held_out')} 张，给智能体看的训练视角 {counts.get('shown_to_agent')} 张）" if counts else ""
        notes.append(f"训练/留出划分：{frame['split']}{extra}。留出视角的照片在生成场景的全过程里没有给过智能体。")
    if checklist:
        bits = [str(checklist.get(k)) for k in ("annotated_by", "note") if checklist.get(k)]
        if bits:
            notes.append(f"交互清单（{cfg.get('checklist')}）：" + "。".join(b.rstrip("。") for b in bits) + "。")
    if cfg.get("references"):
        notes.append("引用点的数字来自外部文献，不是在这台机器上算出来的，评测设置未必和这里逐项一致，只作位置参照。")
    note_html = "".join(f"<li>{_esc(n)}</li>" for n in notes)
    return (f'<footer id="provenance"><h2>来历</h2><div class="runs">{"".join(items)}</div>'
            f'<ul class="legend-list">{note_html}</ul></footer>')


# ---------------------------------------------------------------- 样式和脚本

# 配色取自 dataviz 的参考色板：蓝（flat）、橙（textured）两个类别色在明暗两种底色上都过了
# 全配对的色觉障碍分离度和对比度检查；引用点用中性墨色的空心圆，不占类别色。
_CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#6f6d68;
--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.12);--s1:#2a78d6;--s2:#eb6834;--ok:#008300;--bad:#d03b3b;
--link:#1c5cab;--wash:rgba(11,11,11,.04);--warn-bg:#fdf3d7;--warn-edge:#c98500}
@media (prefers-color-scheme:dark){:root{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--muted:#a3a19a;--grid:#2c2c2a;--axis:#4a4a46;--border:rgba(255,255,255,.14);--s1:#3987e5;
--s2:#d95926;--ok:#0ca30c;--bad:#e66767;--link:#86b6ef;--wash:rgba(255,255,255,.05);--warn-bg:#2b2412;--warn-edge:#c98500}}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--page);color:var(--ink);font:15px/1.6 system-ui,-apple-system,"PingFang SC","Hiragino Sans GB","Microsoft YaHei","Segoe UI",sans-serif;overflow-wrap:anywhere}
main{max-width:1180px;margin:0 auto;padding:24px 16px 56px}
h1{font-size:26px;line-height:1.25;margin:0 0 8px}
h2{font-size:19px;margin:44px 0 8px;padding-top:16px;border-top:1px solid var(--border)}
h3{font-size:15px;margin:20px 0 8px}
p{margin:8px 0}
.lead{color:var(--ink2);max-width:72ch}
.meta,.note{color:var(--muted);font-size:13px}
.note{max-width:80ch}
a{color:var(--link)}
code{font:13px/1.4 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--wash);padding:1px 4px;border-radius:4px}
.sub{display:block;font-weight:400;font-size:12px;color:var(--muted)}
h3 .sub{display:inline;margin-left:8px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch;max-width:100%}
.chart-card{padding:12px 12px 4px;position:relative}
.chart{display:block;width:100%;height:auto;margin:0 auto}
.chart-wide{max-width:860px}
.chart-narrow{display:none;max-width:460px}
@media (max-width:640px){.chart-wide{display:none}.chart-narrow{display:block}}
.chart text{font-size:12px;fill:var(--ink2)}
.chart .tick{fill:var(--muted);font-variant-numeric:tabular-nums}
.chart .axis-title{fill:var(--ink);font-weight:600}
.chart .axis-note{fill:var(--muted)}
.chart .pt-label{fill:var(--ink);paint-order:stroke;stroke:var(--surface);stroke-width:3px;stroke-linejoin:round}
.grid{stroke:var(--grid);stroke-width:1}
.axis{stroke:var(--axis);stroke-width:1}
.link{stroke:var(--axis);stroke-width:2;stroke-linecap:round}
.leader{stroke:var(--muted);stroke-width:1}
.mk-flat{fill:var(--s1);stroke:var(--surface);stroke-width:2}
.mk-textured{fill:var(--s2);stroke:var(--surface);stroke-width:2;stroke-linejoin:round}
.mk-cited{fill:var(--surface);stroke:var(--ink2);stroke-width:2}
.hit{fill:transparent;cursor:pointer;outline:none}
.hit:focus-visible,.hit:hover{stroke:var(--ink);stroke-width:1.5}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink2);padding:2px 4px 10px}
.key{display:inline-flex;align-items:center;gap:6px}
.key svg{width:14px;height:14px;flex:none;overflow:visible}
#tip{position:fixed;z-index:10;pointer-events:none;max-width:min(320px,calc(100vw - 24px));background:var(--surface);color:var(--ink);
border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:13px;line-height:1.5;box-shadow:0 4px 16px rgba(0,0,0,.18)}
#tip[hidden]{display:none}
#tip div:first-child{color:var(--ink2)}
#tip div+div{font-weight:600;font-variant-numeric:tabular-nums}
table{border-collapse:collapse;width:100%;font-size:14px}
table.checks{width:auto;min-width:min(100%,640px)}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:top;white-space:nowrap}
thead th{font-size:13px;color:var(--ink2);font-weight:600;border-bottom:1px solid var(--axis)}
tbody tr:last-child>*{border-bottom:0}
tbody th{font-weight:600}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
td.remark{white-space:normal;min-width:22em;color:var(--ink2);font-size:13px}
td.muted{color:var(--muted)}
tr.cited th,tr.cited td{font-style:normal;background:var(--wash)}
.badge{display:inline-block;margin-left:8px;padding:0 6px;border:1px solid var(--ink2);border-radius:999px;font-size:12px;font-weight:400;color:var(--ink2)}
.ok{color:var(--ok);font-weight:700}
.bad{color:var(--bad);font-weight:700}
.caveat{background:var(--warn-bg);border-left:3px solid var(--warn-edge);padding:8px 12px;border-radius:0 6px 6px 0;max-width:90ch}
.legend-list{margin:6px 0;padding-left:1.3em;color:var(--ink2);font-size:14px;max-width:90ch}
.legend-list li{margin:3px 0}
.view h3{font-weight:500;color:var(--ink2);font-size:13px;margin:18px 0 6px}
.strip{display:flex;gap:10px;padding-bottom:8px}
figure{margin:0}
.strip figure{flex:0 0 auto;width:clamp(270px,calc((100% - (var(--n,3) - 1)*10px)/var(--n,3)),520px)}
figure img{display:block;width:100%;height:auto;border-radius:6px;background:var(--wash)}
.missing{display:flex;align-items:center;justify-content:center;aspect-ratio:3/2;border:1px solid var(--border);border-radius:6px;color:var(--muted);font-size:13px}
figcaption{display:flex;flex-wrap:wrap;justify-content:space-between;gap:2px 10px;font-size:13px;padding:5px 2px 0}
figcaption .who{color:var(--ink)}
figcaption .metric{color:var(--ink2);font-variant-numeric:tabular-nums}
.swatches{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.swatch figcaption{display:block}
.swatch figcaption span{display:block}
.swatch .prompt{color:var(--ink2)}
.runs{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px}
.run{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:10px 14px}
.run h3{margin:0 0 6px}
dl{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:0;font-size:13px}
dt{color:var(--muted);white-space:nowrap}
dd{margin:0}
"""

# 悬停提示：内容从 data-tip 里按行读，用 textContent 写进去，不拼 HTML。
_JS = """
(function(){var tip=document.getElementById('tip');if(!tip)return;
function show(el,x,y){tip.textContent='';el.getAttribute('data-tip').split('\\n').forEach(function(line){
var d=document.createElement('div');d.textContent=line;tip.appendChild(d);});tip.hidden=false;
var w=tip.offsetWidth,h=tip.offsetHeight,vw=document.documentElement.clientWidth;
var left=Math.min(Math.max(8,x+14),vw-w-8),top=y-h-12;if(top<8)top=y+16;
tip.style.left=left+'px';tip.style.top=top+'px';}
function hide(){tip.hidden=true;}
Array.prototype.forEach.call(document.querySelectorAll('.hit'),function(el){
el.addEventListener('pointermove',function(e){show(el,e.clientX,e.clientY);});
el.addEventListener('pointerleave',hide);
el.addEventListener('focus',function(){var r=el.getBoundingClientRect();show(el,r.left+r.width/2,r.top);});
el.addEventListener('blur',hide);});
window.addEventListener('scroll',hide,{passive:true});})();
"""


# ---------------------------------------------------------------- 入口

def build_report(cfg: dict, pack_dir: Path, run_dirs: list[Path], out_dir: Path) -> Path:
    """把 run_dirs 里的结果汇成 out_dir/index.html（图在 out_dir/img/），返回 index.html 的路径。

    cfg 是读进来的 yaml 字典；用到 scene、data.image_dir、references 和 checklist
    （相对仓库根目录的路径）。一次运行或多次运行、只有 flat 或两种模式都有、
    有没有引用点、有没有 LPIPS，都能出页面。
    """
    pack_dir, out_dir = Path(pack_dir), Path(out_dir)
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    runs = [_load_run(Path(d), i) for i, d in enumerate(run_dirs)]
    points = [p for run in runs for p in run["points"]]
    _dedupe_names(points)
    points += _load_references(cfg)

    views = (_read_json(pack_dir / "eval" / "cameras.json", {}) or {}).get("views") or []
    views = [v for v in views if isinstance(v, dict)]
    frame = _read_json(pack_dir / "frame.json")
    checklist = _read_json(REPO / str(cfg["checklist"])) if cfg.get("checklist") else None
    if not isinstance(checklist, dict):
        checklist = None
    image_dir = str((cfg.get("data") or {}).get("image_dir") or (frame or {}).get("image_dir") or "未知")

    body = "".join([
        _header(cfg, views, image_dir),
        _chart_section(points),
        _summary_table(points),
        _gallery(pack_dir, views, points, img_dir),
        _checklist_section(runs, checklist),
        _textures_section(runs, img_dir),
        _provenance(runs, frame if isinstance(frame, dict) else None, checklist, cfg),
    ])
    page = (
        '<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>两轴评测 · {_esc(cfg.get("scene", ""))}</title><style>{_CSS}</style></head>'
        f'<body><main>{body}</main><div id="tip" role="status" hidden></div><script>{_JS}</script></body></html>\n')
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
