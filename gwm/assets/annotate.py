"""给程序里的 asset 节点挑一个具体资产，结果写进 `geom.ref`。

分工：模型（或直译）只负责说「这里有一个火车那么大的东西」，也就是 `query` 和 `extent`；
挑哪一个具体模型是这一步做的，因为它需要看画面，而模型那一步手上只有文字证据。
`ref` 是事后标注，不是模型要预测的字段——训练数据里不带它。

挑的办法是文字分数和图像分数加权：文字来自资产名与 kit 名，图像来自把视频里这个物体
抠出来、和资产缩略图比 DINOv2 相似度。只有文字的时候查 "toy train" 会选中
`train-connector` 这种零件，加上图像才分得出哪个像视频里那辆车。
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .retrieve import retrieve

# 缩略图的背景色，抠图时用同一个值填充，免得模型把背景差异当成形状差异
THUMB_BG = (242, 242, 242)


def bbox_of(observation: dict) -> tuple[list[float], list[int] | None] | None:
    """取出 (xyxy 四个数, 这个框所在的图像尺寸)。

    证据里的 bbox 是 `{format, values, space, image_size}` 这种带单位的结构，而不是裸的
    四元组，而且它的坐标是在**掩码那一档分辨率**上的（几何后端把帧缩到 14 的倍数），
    和原帧不是一个尺寸。拿它直接去裁原帧会裁错地方，所以这里把尺寸一并带出去。
    """
    box = observation.get("bbox")
    if isinstance(box, dict):
        values, size = box.get("values"), box.get("image_size")
        if box.get("format") not in (None, "xyxy") or not (isinstance(values, (list, tuple)) and len(values) == 4):
            return None
        return [float(v) for v in values], (list(size) if isinstance(size, (list, tuple)) and len(size) == 2 else None)
    if isinstance(box, (list, tuple)) and len(box) == 4:
        return [float(v) for v in box], None
    return None


def best_observation(obj: dict) -> dict | None:
    """挑这个物体看得最清楚的一帧：框的面积最大的那一帧。"""
    best, best_area = None, -1.0
    for ob in obj.get("observations", []):
        parsed = bbox_of(ob)
        if parsed is None:
            continue
        box = parsed[0]
        area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
        if area > best_area:
            best, best_area = ob, area
    return best


def crop_object(frame_rgb: np.ndarray, mask: np.ndarray, bbox, pad: float = 0.12):
    """按掩码把物体抠到中性背景上，再按包围盒裁剪。失败返回 None。

    必须抠掉背景：资产缩略图是干净底色上的单个物体，拿一整块带房间背景的裁剪去比，
    比出来的是「这两张图的背景像不像」。
    """
    height, width = frame_rgb.shape[:2]
    if mask.shape != (height, width):
        from ..perception.masks import resize_mask
        mask = resize_mask(mask, (width, height))
    if not mask.any():
        return None
    composited = np.where(mask[:, :, None], frame_rgb, np.array(THUMB_BG, dtype=frame_rgb.dtype))
    x0, y0, x1, y1 = (float(v) for v in bbox)
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    x0, y0 = max(0, int(x0 - dx)), max(0, int(y0 - dy))
    x1, y1 = min(width, int(x1 + dx)), min(height, int(y1 + dy))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return composited[y0:y1, x0:x1]


def load_shape_cache(cfg: dict | None = None) -> dict:
    """资产 id → 边长比例（三边排序后除以最长边）。没有缓存就返回空，行为不变。

    由 `scripts/build_asset_shape_cache.py` 预计算。路径从配置
    `assets.shape_cache` 读，其次环境变量 GWM_ASSET_SHAPES。
    """
    import os
    path = ((cfg or {}).get("assets") or {}).get("shape_cache") or os.environ.get("GWM_ASSET_SHAPES")
    if not path:
        return {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {k: v for k, v in data.items() if isinstance(v, list) and len(v) == 3}


def observed_edge_ratios(obj: dict | None) -> list[float] | None:
    """观测到的边长比例，口径和资产缓存**完全一致**：三边排序后除以最长边。

    两边口径不一致的话这个重排就是在比噪声。资产那边是轴对齐包围盒的三边，
    这边取逐帧包围盒尺寸的中位数——都是「这个物体三个方向上各有多长」。
    """
    obbs = (obj or {}).get("obb") or []
    sizes = [b.get("size") for b in obbs if isinstance(b.get("size"), list) and len(b["size"]) == 3]
    if not sizes:
        return None
    med = np.median(np.asarray(sizes, float), axis=0)
    if not np.isfinite(med).all() or med.max() <= 1e-9:
        return None
    return [float(v) for v in np.sort(med)[::-1] / med.max()]


def shape_rerank(hits: list, observed: list[float] | None, shapes: dict,
                 weight: float = 0.5) -> list:
    """按「资产的边长比例和观测差多少」重排候选。

    **为什么这是创新点一在管线里的落点。** `shape_kind_accuracy` 结构性恒零
    （`effective_shape()` 对解析到资产的节点返回 "mesh"，而真值全是基本体），
    所以它量不出任何改进。真正有数的是 `shape_ok`——它比的正是**三条边的比例**。
    而资产检索原来只看类名和图像，`Asset` 里一个几何字段都没有：**观测到的包围盒
    比例这个纯几何量，完全没参与「选哪个资产」这个离散选择**。

    这就是「用连续稠密表征去监督离散的符号程序」最直接的一个实例。

    只重排、不改收不收。`retrieve()` 的注释写明文字分决定收不收、图像分只在候选里
    重排，理由是量纲不同、混进阈值会误杀。形状分同理：它是个 0~1 的比例距离，
    和文字分不可比，所以只参与排序。

    观测拿不到比例（没有 obb）时原样返回，行为不变。
    """
    if not observed or not shapes or len(hits) < 2:
        return hits
    obs = np.asarray(observed, float)

    def distance(asset_id: str) -> float:
        r = shapes.get(asset_id)
        return 1.0 if r is None else float(np.abs(np.asarray(r, float) - obs).mean())

    scored = [(a, s, distance(a.id)) for a, s in hits]
    best_text = max((s for _a, s, _d in scored), default=1.0) or 1.0
    scored.sort(key=lambda t: (-((1 - weight) * (t[1] / best_text) + weight * (1 - t[2])), t[0].id))
    return [(a, s) for a, s, _d in scored]


def residual_pick(clouds: list, candidates: list, root, top: int = 0,
                  n_views: int = 7) -> tuple:
    """**用配准残差从候选里挑**：把每个候选网格缩到观测点云的尺度，ICP 配一次，
    取残差最小的。返回 (资产, 残差) 或 (None, None)。

    这是「用连续稠密表征监督离散符号选择」的落地形式。梯度传不过去不要紧——
    「哪个网格配得上这团点」这个判据本来就不需要梯度。

    实测（assetgt60，74 个带真值 ref 的物体，按样本加权）：

        真值 + 32 个随机干扰          挑对 0.284（随机猜 0.030，9.4 倍）
        真值 + 32 个**形状最接近**的干扰 挑对 **0.243**（随机猜 0.030，**8.0 倍**）

    第二行是密度匹配的零假设——随机资产长得和目标完全不像，而真实候选集个个都像。
    换成难干扰项只掉了 0.04，说明残差不是靠排除「长得完全不像的」在赢。
    真值静止的物体上难干扰项 33 选 1 挑对 0.500。

    **尺度要缩到逐帧包围盒的中位对角线上，不能用并帧的。** 并帧点云随视角扫动，
    实测比逐帧中位大 **1.89 倍**，而受控扫描说模板相对点云大 1.2 倍就越过对照、
    1.3 倍崩掉。这个坑踩过一次（资产模板的角度 RMSE 80.92°，改掉之后 32.54°）。
    偏小是安全的，所以宁可缩小一点。

    **已知边界：物体在转的时候这条就垮**（periodic_rotate 挑对 0.000、spin 0.125，
    而静止 0.500）。调用方拿不准的时候该退回文字那条，而不是硬挑。
    """
    import numpy as _np
    from ..perception.registration import icp
    from .mesh import load_points

    if not clouds or not candidates:
        return None, None
    per_frame = [float(_np.linalg.norm(_np.asarray(c, float).max(0) - _np.asarray(c, float).min(0)))
                 for c in clouds if len(_np.asarray(c, float)) >= 10]
    if not per_frame:
        return None, None
    obs_diag = float(_np.median(per_frame))
    # **用几帧，不是一帧。** 一帧只是一个局部视角，同一包围盒比例下完全不同的物体
    # 在单个视角上可能都配得差不多。实测（33 个形状最接近的候选里挑对的比例）：
    #
    #     1 帧 0.149 ／ 3 帧 0.243 ／ **7 帧 0.297** ／ 15 帧 0.284 ／ 全部 0.311
    #
    # 平台在 7 帧，静止物体上 0.625。另外**起点不能放在第 0 帧**：只用首帧 0.149、
    # 只用中间帧 0.243，首帧的可见面最偏。均匀抽时起点放到半个间隔处，避开两端。
    #
    # 已试过并否掉的：绕竖轴加 4/8 个初始偏航、每帧取最小残差，反而从 0.297 掉到
    # 0.216——取最小等于给每个候选更大的搜索预算，而错的候选从中得益更多。
    # 「挑哪个」这类任务里，候选之间的分必须可比。
    if 0 < n_views < len(clouds):
        stride = len(clouds) / float(n_views)
        picks = [clouds[min(len(clouds) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    else:
        picks = clouds
    queries = [_np.asarray(c, float) for c in picks if len(_np.asarray(c, float)) >= 30]
    if not queries:
        return None, None
    best, best_rmse = None, float("inf")
    for asset in (candidates[:top] if top > 0 else candidates):
        try:
            pts = load_points(Path(root) / asset.path, samples=3000)
        except (OSError, ValueError):
            continue
        if pts is None or len(pts) < 10:
            continue
        diag = float(_np.linalg.norm(pts.max(0) - pts.min(0)))
        if diag <= 1e-9 or obs_diag <= 1e-9:
            continue
        model = (pts - pts.mean(0)) * (obs_diag / diag)
        rs = []
        for q in queries:
            step = icp(q, model)
            if step["ok"] and step["rmse"] is not None:
                rs.append(float(step["rmse"]))
        if not rs:
            continue
        # 取中位数不取均值：某一帧掩码漏了一块就会拉出离谱的残差。
        score = float(_np.median(rs))
        if score < best_rmse:
            best_rmse, best = score, asset
    return best, (best_rmse if best is not None else None)


def _project_outline(pts, R_wc, yaw: float, grid: int = 96):
    """把候选网格按偏航转一下、用相机朝向转进相机系、正交投影成二值轮廓。

    **正交投影是关键**：透视需要知道物体离相机多远，而那个距离正是深度给的，
    而深度是坏的（实测 VGGT 沿相机视线把物体压扁到最短轴的 0.526）。
    正交投影不需要它，物体只占几百到几千像素时形变也可忽略。
    """
    import numpy as _np

    a = _np.asarray(pts, float)
    c, sn = _np.cos(yaw), _np.sin(yaw)
    R_yaw = _np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])
    uv = ((a @ R_yaw.T) @ _np.asarray(R_wc, float).T)[:, :2]
    lo, hi = uv.min(0), uv.max(0)
    span = _np.maximum(hi - lo, 1e-9)
    k = (grid - 2) / span.max()
    ij = _np.floor((uv - lo) * k).astype(int)
    ij[:, 0] += (grid - 1 - int(span[0] * k)) // 2
    ij[:, 1] += (grid - 1 - int(span[1] * k)) // 2
    ij = _np.clip(ij, 0, grid - 1)
    out = _np.zeros((grid, grid), bool)
    out[ij[:, 1], ij[:, 0]] = True
    return out


def _largest_component(m):
    """只留掩码的最大连通块。

    **不碰候选，所以没有循环依赖。** 「用挑中的网格去修掩码」那条有陷阱——修出来
    的掩码被候选塑形，再挑必然偏向同一个候选（配对条件含被测量）。取最大连通块
    完全独立于候选。

    为什么直指问题：神经掩码的**形状统计和真值几乎一样**（占外接框 0.750 对 0.790、
    边界复杂度 25.3 对 24.9），却一个都没命中——说明差别不在形状质量，在**框的是
    什么东西**。碎片和粘连都会在连通性上露出来。
    """
    import numpy as _np

    try:
        from scipy.ndimage import label
    except ImportError:
        return m
    a = _np.asarray(m, bool)
    if not a.any():
        return a
    lab, n = label(a)
    if n <= 1:
        return a
    counts = _np.bincount(lab.ravel())
    counts[0] = 0
    return lab == int(counts.argmax())


def _mask_outline(mask, grid: int = 96, largest_only: bool = False):
    """观测掩码按它的二维包围盒等比归一化到同一个格子。"""
    import numpy as _np

    m = _np.asarray(mask, bool)
    if largest_only:
        m = _largest_component(m)
    ys, xs = _np.nonzero(m)
    if len(xs) < 20:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    k = (grid - 2) / float(max(w, h))
    ij = _np.stack([((xs - x0) * k).astype(int), ((ys - y0) * k).astype(int)], axis=1)
    ij[:, 0] += (grid - 1 - int(w * k)) // 2
    ij[:, 1] += (grid - 1 - int(h * k)) // 2
    ij = _np.clip(ij, 0, grid - 1)
    out = _np.zeros((grid, grid), bool)
    out[ij[:, 1], ij[:, 0]] = True
    return out


def silhouette_pick(masks_per_frame: dict, cams: dict, candidates: list, root,
                    top: int = 0, n_views: int = 3, n_yaw: int = 8,
                    frame_agg: str = "median",
                    largest_component: bool = False,
                    text_scores: dict | None = None,
                    image_scores: dict | None = None) -> tuple:
    """**用轮廓 IoU 从候选里挑**，全程不碰深度。返回 (资产, IoU) 或 (None, None)。

    为什么换掉深度残差。根因是 VGGT 沿相机视线把物体压扁（最短轴只剩正确比例的
    0.526），而**沿视线的误差投影到像平面上恰好是零**。实测（真实几何，
    33 个形状最近邻候选，随机 0.030）：

        ICP 残差（三维表面，用深度）      0.041   1.4 倍
        按观测位姿渲染轮廓 + IoU          **0.108**  **3.6 倍**

    而这套方法的真实形态（图像提候选 + 轮廓挑，候选里没真值时记失败）：

        图像前 64 名   覆盖率 0.284   挑对 **0.054**   条件准确率 0.19
        图像前 256 名  覆盖率 0.419   挑对  0.014      条件准确率 0.033

    **候选集要小**：前 64 名比前 256 名好近四倍，尽管后者覆盖率高一半——
    挑的能力随候选数衰减得比覆盖率涨得快。

    每个候选给同样的 n_yaw 个偏航、n_views 帧，预算严格相同，所以「取最好」
    不会偏袒错的候选（预算不等时会，这个坑量到过）。
    """
    import numpy as _np
    from .mesh import load_points

    if not masks_per_frame or not candidates:
        return None, None
    avail = [f for f in sorted(masks_per_frame) if f in cams]
    if not avail:
        return None, None
    if 0 < n_views < len(avail):
        stride = len(avail) / float(n_views)
        avail = [avail[min(len(avail) - 1, int(stride * (i + 0.5)))] for i in range(n_views)]
    obs = [(f, _mask_outline(masks_per_frame[f], largest_only=largest_component))
           for f in avail]
    obs = [(f, g) for f, g in obs if g is not None]
    if not obs:
        return None, None
    yaws = [2.0 * _np.pi * j / n_yaw for j in range(n_yaw)]
    scored = []
    for asset in (candidates[:top] if top > 0 else candidates):
        try:
            tpl = load_points(Path(root) / asset.path, samples=20000)
        except (OSError, ValueError):
            continue
        if tpl is None or len(tpl) < 50:
            continue
        tpl = tpl - tpl.mean(0)
        per = []
        for f, g in obs:
            R_wc = _np.asarray(cams[f], float)[:3, :3].T
            hi = 0.0
            for y in yaws:
                sil = _project_outline(tpl, R_wc, float(y))
                union = float(_np.logical_or(sil, g).sum())
                if union > 0:
                    hi = max(hi, float(_np.logical_and(sil, g).sum()) / union)
            per.append(hi)
        if per:
            # **跨帧怎么合，决定了多帧是「重采样」还是「约束」。**
            # 原来用中位：把每帧当成同一个测试的独立样本取稳健平均，
            # 所以**加帧只是重采样**。实测 3→7 帧三个指标里两个变差
            # （命中 0.213→0.177、形状距离 0.0798→0.0925）。
            #
            # 取**最小值**才是多视角**约束**：候选在任何一帧上对不上就被否掉。
            # 错的候选常能在某个视角下凑合匹配，但很难所有视角都匹配。
            # residual_pick 那边多视角真有收益，靠的是累积表面覆盖，不是平均。
            scored.append((float(_np.min(per) if frame_agg == "min"
                                 else _np.median(per)), asset))
    if not scored:
        return None, None
    # **倒数名次融合**，不是只按 IoU 挑。
    #
    # 候选是图像检索排出来的（正确资产在静止物体上中位排第 36），而只按 IoU 重排
    # 等于把那个排序信息整个扔掉。两个信号都弱但**独立**：图像分看外观，轮廓看
    # 二维形状。RRF 无参数，而且对两边的分数量纲都不敏感——今天量过文字分和图像分
    # 的量纲其实可比，但 IoU 和余弦相似度不可比，所以按名次融合比按分数融合稳。
    #
    # k=60 是 RRF 的常用值；名次从 1 起算。
    by_iou = {a.id: i for i, (_v, a) in enumerate(sorted(scored, key=lambda t: -t[0]))}
    order_in = {a.id: i for i, a in enumerate(candidates[:top] if top > 0 else candidates)}
    # **三路都用名次。** 原来是两步混的：retrieve() 先用**分数**把文字和图像混成
    # 一个顺序（order_in），这里再用名次把轮廓 IoU 和那个顺序融合。
    # 而实测文字分和图像分的量纲可比（中位 0.504 对 0.501），但 **IoU 和它们完全
    # 不可比**——所以分数融合那一步是把两种可比的和一种不可比的混在一起。
    # 给了 text_scores / image_scores 就三路各自排名次再 RRF，去掉最后一处分数混合。
    ids = [a.id for _v, a in scored]
    def _rank(d):
        if not d:
            return None
        order = sorted(ids, key=lambda i: -float(d.get(i, 0.0)))
        return {i: k for k, i in enumerate(order)}
    r_text, r_img = _rank(text_scores), _rank(image_scores)
    best, best_rrf, best_iou = None, -1.0, None
    for iou, asset in scored:
        rrf = 1.0 / (60.0 + by_iou.get(asset.id, len(scored)) + 1)
        if r_text is None and r_img is None:
            rrf += 1.0 / (60.0 + order_in.get(asset.id, len(order_in)) + 1)
        else:
            if r_text is not None:
                rrf += 1.0 / (60.0 + r_text.get(asset.id, len(ids)) + 1)
            if r_img is not None:
                rrf += 1.0 / (60.0 + r_img.get(asset.id, len(ids)) + 1)
        if rrf > best_rrf:
            best_rrf, best, best_iou = rrf, asset, iou
    return best, (best_iou if best is not None else None)


CROP_NAMER_PROMPT = (
    "You see one object cut out of a 3D scene. Reply with a JSON object "
    '{"name": "..."} giving the most specific noun phrase that identifies this '
    "object as a 3D asset would be named: include the subtype and distinguishing "
    "detail, 2-4 words, lowercase. Examples of the level of detail wanted: "
    "\"bunk bed\", \"sports sedan\", \"pine tree\", \"garage door\", "
    "\"waterfall rock\", \"corner wall panel\". Do not answer with a bare "
    "category like \"object\", \"box\" or \"block\"."
)


def name_crop(client, crop) -> str:
    """让 VLM 看**单个物体的裁剪图**说出具体名字，当检索的查询词。

    **为什么是这一步。** 物体级指标量下来：神经分割的 `object_recall` 是 **0.901**
    ——检测器找得到这些物体，过检也只是中等（精确率 0.683）。真正分开两臂的是
    `class_accuracy`：神经 **0.000** 对真值分割 0.170。**物体找到了，但贴给它的
    类名是错的**，而类名就是文字检索的查询词 → 候选集里没有正确答案 → 0 命中。

    此前的命名器是**整帧**级的，产出的短语拿去做检测；这里是**单物体**级的，
    产出的名字拿去做检索。两件事不一样，后者此前一次都没试过。

    提示词明确要「子类 + 区分性细节」，并禁掉 object/box/block 这种裸类别——
    因为 `prompts/namer.md` 里那句 `no colours, no adjectives about size`
    正好禁掉了今天量到值钱的那一维（真值类名里管用的是 sedan sports、tree pine、
    candy cane red 这种）。
    """
    import json as _json

    schema = {"type": "object", "properties": {"name": {"type": "string"}},
              "required": ["name"]}
    try:
        from PIL import Image
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            Image.fromarray(crop).save(tmp.name, quality=92)
            path = tmp.name
        try:
            r = client.chat(CROP_NAMER_PROMPT, "Object attached.", images=[path],
                            json_schema=schema, temperature=0.2)
        finally:
            os.unlink(path)
        name = str((r.get("json") or {}).get("name", "")).strip().lower()
    except Exception:
        return ""
    # 裸类别挡掉：它们正是现在失败的样子，放过去等于没改
    if name in {"object", "box", "block", "thing", "item", "shape", "part", ""}:
        return ""
    return name[:60]


def choose_assets(program: dict, evidence: dict, frames: list[dict], masks: dict,
                  catalog: list, index=None, embedder=None,
                  image_weight: float = 0.6, min_score: float = 0.45,
                  top_k: int = 5, shapes: dict | None = None,
                  shape_weight: float = 0.0, image_admit: int = 0,
                  image_gate_rank: int = 0, clouds: dict | None = None,
                  residual_top: int = 0, root=None,
                  silhouette_top: int = 0, masks_by_obj: dict | None = None,
                  cams: dict | None = None,
                  mask_largest_component: bool = False,
                  crop_namer=None, three_way_rrf: bool = False,
                  gray_query: bool = False, silhouette_views: int = 3,
                  silhouette_agg: str = "median") -> list[dict]:
    """就地给每个 asset 节点填上 `geom.ref`，返回每个决策的记录。"""
    from PIL import Image

    # 程序里的 id 是证据 id 过了一遍 slug 的（会截断、会换掉非法字符），所以两边
    # 不能直接对上。两种写法都登记进来，掩码仍然按证据 id 取——masks.npz 的键是证据 id。
    from ..taxonomy import slug
    by_id = {}
    for obj in evidence.get("objects", []):
        by_id[obj["id"]] = obj
        by_id.setdefault(slug(obj["id"]), obj)
    frame_by_index = {f.get("index"): f for f in frames}
    decisions = []
    for node in program.get("objects", []):
        geom = node.get("geom", {})
        if geom.get("kind") != "asset":
            continue
        query = geom.get("query") or node.get("class") or ""
        record = {"id": node["id"], "query": query, "image_used": False}
        record["query_source"] = "program"

        image_scores = None
        crop = None            # 只有走到图像那一路才会赋值；裁剪图命名也要用它
        obj = by_id.get(node["id"])
        observation = best_observation(obj) if obj else None
        # 走不到图像这一路的时候要说清楚卡在哪一步。只写一个「没用图像」的话，
        # id 对不上和索引没加载这两种完全不同的毛病看起来一模一样。
        if index is None or embedder is None:
            record["image_skipped"] = "no_index"
        elif obj is None:
            record["image_skipped"] = f"no_evidence_object_for_{node['id']}"
        elif observation is None:
            record["image_skipped"] = "no_observation_with_bbox"
        if observation is not None and index is not None and embedder is not None:
            frame = frame_by_index.get(observation.get("frame_index"))
            mask = (masks.get(obj["id"]) or {}).get(observation.get("frame_index"))
            if not frame:
                record["image_skipped"] = f"no_frame_{observation.get('frame_index')}"
            elif mask is None:
                record["image_skipped"] = f"no_mask_for_{obj['id']}@{observation.get('frame_index')}"
            if frame and mask is not None:
                try:
                    rgb = np.asarray(Image.open(frame["file"]).convert("RGB"))
                    box, box_size = bbox_of(observation)
                    # 框是在掩码那一档分辨率上给的，按宽度比例缩放到原帧坐标系
                    if box_size and box_size[0]:
                        k = rgb.shape[1] / float(box_size[0])
                        box = [v * k for v in box]
                    crop = crop_object(rgb, np.asarray(mask), box)
                    if crop is not None:
                        _q = Image.fromarray(crop)
                        if gray_query:
                            # 两侧都去掉颜色。数据集给每个物体重新上色，而库里
                            # 是资产原生外观；留一视角自匹配证明嵌入本身没问题
                            # （前 64 覆盖 0.990），差的是这个域。
                            _q = _q.convert("L").convert("RGB")
                        vector = embedder.embed_images([_q])[0]
                        image_scores = index.scores(vector)
                        record["image_used"] = True
                        record["frame_index"] = observation.get("frame_index")
                    record["evidence_id"] = obj["id"]
                except (OSError, ValueError, KeyError) as error:
                    record["image_error"] = f"{type(error).__name__}: {error}"[:160]

        # 残差挑资产要在**大候选集**上做才有意义：图像嵌入把正确资产放在中位
        # 第 365 名，前 5 个里基本不会有它。所以开了残差就把 k 放大到 residual_top。
        # **两个开关都要算进去。** 第一版只看 residual_top，而轮廓挑那条配的是
        # silhouette_top，于是 k 退回默认的 5——跑的不是「图像前 64 名里挑」，
        # 是「文字前 5 名里挑」。冒烟时记录的候选名次全在 0~4，才露了馅。
        _wide = max(residual_top, silhouette_top)
        _k = max(top_k, _wide) if _wide > 0 else top_k
        # **用裁剪图重新命名**，再拿新名字当检索查询词。检测没问题（召回 0.901），
        # 坏的是贴给物体的类名（class_accuracy 0.000），而类名就是查询词。
        if crop_namer is not None and crop is not None:
            _n = name_crop(crop_namer, crop)
            if _n:
                record["crop_name"] = _n
                record["query_source"] = "crop_namer"
                query = _n
                record["query"] = _n
        hits = retrieve(query, catalog, k=_k, image_scores=image_scores,
                        image_weight=image_weight if image_scores else 0.0,
                        image_admit=image_admit)
        # 形状重排：把观测到的边长比例用进「选哪个资产」这个离散选择。
        # shape_weight 为 0（默认）时完全不动，行为和以前一致。
        if shape_weight > 0 and shapes:
            obs_ratio = observed_edge_ratios(obj)
            if obs_ratio:
                hits = shape_rerank(hits, obs_ratio, shapes, weight=shape_weight)
                record["shape_reranked"] = True
                record["observed_ratio"] = [round(v, 3) for v in obs_ratio]
        # **用配准残差从候选里挑。** 这是「连续稠密表征监督离散符号选择」的落地形式：
        # 文字和图像只负责**提候选**，谁配得上这团点由几何说了算。见 residual_pick。
        # **轮廓挑**优先于残差挑：真实几何下 0.108 对 0.041（随机 0.030）。
        _oid = obj["id"] if obj else None
        _mk = (masks_by_obj or {}).get(_oid) if _oid else None
        if silhouette_top > 0 and _mk and cams and root is not None and hits:
            _pick, _iou = silhouette_pick(
                _mk, cams, [a for a, _s in hits], root,
                top=silhouette_top, largest_component=mask_largest_component,
                n_views=silhouette_views, frame_agg=silhouette_agg,
                text_scores=({a.id: sc for a, sc in hits} if three_way_rrf else None),
                image_scores=(image_scores if three_way_rrf else None))
            if _pick is not None:
                _by_id = {a.id: sc for a, sc in hits}
                record["silhouette_iou"] = round(float(_iou), 4)
                record["silhouette_rank"] = next(
                    (i for i, (a, _s) in enumerate(hits) if a.id == _pick.id), None)
                hits = [(_pick, _by_id.get(_pick.id, 0.0))] + [
                    pr for pr in hits if pr[0].id != _pick.id]
        _cloud = (clouds or {}).get(obj["id"]) if obj else None
        if silhouette_top <= 0 and residual_top > 0 and _cloud and root is not None and hits:
            _pick, _rmse = residual_pick(_cloud, [a for a, _s in hits], root,
                                         top=residual_top)
            if _pick is not None:
                _by_id = {a.id: s for a, s in hits}
                record["residual_rmse"] = round(float(_rmse), 5)
                record["residual_rank"] = next(
                    (i for i, (a, _s) in enumerate(hits) if a.id == _pick.id), None)
                hits = [(_pick, _by_id.get(_pick.id, 0.0))] + [
                    pair for pair in hits if pair[0].id != _pick.id]
        record["candidates"] = [{"asset_id": a.id, "score": s} for a, s in hits[:5]]
        # 图像那一路的最高分记进**决策记录**，不进程序。程序是一份受校验的契约：
        # schema 里 assetGeom 的 additionalProperties 是 false，往 geom 里塞一个
        # ref_image_score 会让每个资产节点都校验失败——试过，十二条片段只剩两条能跑完，
        # 资产覆盖率掉到 0。诊断信息归 manifest，不归程序。
        if image_scores is not None:
            try:
                record["image_score_best"] = round(float(max(image_scores.values())), 4)
            except (AttributeError, ValueError, TypeError):
                pass
        # 收不收这个资产。原来只有一道门：**文字分**过 min_score。
        #
        # 图像塞进来的候选文字分恒为 0，所以就算融合之后排第一也必然被这道门挡掉、
        # 退回基本体——「图像进候选」这个开关在这道门下面从原理上不可能兑现。
        # 所以图像这一路要有自己的门，而且不能是拍脑袋的绝对阈值（DINOv2 比真实
        # 照片和低多边形渲染，余弦值天然低，量纲和文字分对不上，这是当初不混进
        # 阈值的理由，仍然成立）。
        #
        # 用**本次裁剪图在全库里的名次**当判据：名次是自归一化的，不受这一张图
        # 整体相似度高低的影响。标定来自实测——本片段真值资产的中位名次 365、
        # 进前 32 名 0.176，而别片段集合的零假设是 1117 / 0.031。名次越靠前，
        # 富集倍数越高。
        chosen_pair = hits[0] if hits else None
        record["chosen"] = None
        if chosen_pair is not None:
            top_asset, top_text = chosen_pair
            passed = top_text >= min_score
            # 文字分为 0 说明这个候选是图像塞进来的——自检要数它，否则又会出现
            # 「旋钮没点着却以为测到了」那种事。
            record["admitted_by_image"] = bool(image_scores) and top_text <= 0.0
            if not passed and image_scores and image_gate_rank > 0:
                own = float(image_scores.get(top_asset.id, 0.0))
                rank = 1 + sum(1 for v in image_scores.values() if v > own)
                record["image_gate_rank"] = rank
                passed = rank <= image_gate_rank
            if passed:
                geom["ref"] = top_asset.id
                record.update(chosen=top_asset.id, score=top_text)
        decisions.append(record)
    return decisions
