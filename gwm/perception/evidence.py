"""Evidence extraction: unproject masks with depth -> per-frame OBBs -> smoothing -> screw-motion classification -> contacts -> ground alignment -> evidence.json (+ overlay video)."""
from __future__ import annotations
import copy, json, math, time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from ..taxonomy import is_static_class
from .base import CV2THREE, Geometry, Tracks
from .motion import estimate_motion
from .color import observe_colors
from .registration import infer_articulation, looks_static, track_rigid
from .visibility import checked_visibility


def _quat_xyzw(mat3: np.ndarray) -> list[float]:
    return [float(v) for v in R.from_matrix(mat3).as_quat()]  # scipy: x,y,z,w

# ---------------- geometry helpers ----------------
def unproject(depth: np.ndarray, K: np.ndarray, mask: np.ndarray | None = None, stride: int = 1):
    H, W = depth.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    d = depth[::stride, ::stride]
    valid = d > 1e-4
    if mask is not None: valid &= mask[::stride, ::stride]
    xs, ys, d = xs[valid].astype(np.float64), ys[valid].astype(np.float64), d[valid].astype(np.float64)
    X = (xs - K[0, 2]) / K[0, 0] * d; Y = (ys - K[1, 2]) / K[1, 1] * d
    return np.stack([X, Y, d], axis=1)  # camera coords (OpenCV)

def to_world(pts_cam: np.ndarray, c2w: np.ndarray) -> np.ndarray:
    return (c2w[:3, :3] @ pts_cam.T).T + c2w[:3, 3]

def fit_plane_ransac(pts: np.ndarray, thresh: float, iters: int = 300, rng=None) -> tuple[np.ndarray, float, np.ndarray]:
    """Return (normal, offset d with n.x + d = 0, inlier mask). Largest support."""
    rng = rng or np.random.default_rng(0)
    best = (None, 0.0, np.zeros(len(pts), bool)); n = len(pts)
    if n < 50: return best
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False); p0, p1, p2 = pts[idx]
        nrm = np.cross(p1 - p0, p2 - p0); nn = np.linalg.norm(nrm)
        if nn < 1e-9: continue
        nrm /= nn; d = -nrm @ p0
        inl = np.abs(pts @ nrm + d) < thresh
        if inl.sum() > best[2].sum(): best = (nrm, d, inl)
    if best[0] is None: return best
    # refine with SVD on inliers
    P = pts[best[2]]; c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); nrm = vt[-1]; d = -nrm @ c
    inl = np.abs(pts @ nrm + d) < thresh
    return nrm, float(d), inl

def rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix that maps unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = float(a @ b)
    if np.linalg.norm(v) < 1e-9: return np.eye(3) if c > 0 else R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))

# ---------------- world alignment ----------------
def align_world(geom: Geometry, tracks: Tracks, cfg: dict, rng=None) -> tuple[np.ndarray, float, dict]:
    """Find the ground plane from static pixels, return (T 4x4 applied to world points, scale, info).
    Candidate planes are extracted iteratively; the ground is the large, roughly horizontal plane below the camera
    that most object centres lie above (this rejects sky/far planes). Result world: y up, ground at y=0."""
    pc = cfg["perception"]
    W, H = geom.size()
    pts_all, obj_pts = [], []
    step = max(1, len(geom.frame_indices) // 6)
    for k in range(0, len(geom.frame_indices), step):
        fi = geom.frame_indices[k]
        depth = geom.depth[k]; med = np.median(depth[depth > 1e-4]) if (depth > 1e-4).any() else 1.0
        near = (depth > 1e-4) & (depth < pc.get("far_depth_factor", 3.0) * med)   # drop sky / far garbage
        static_mask = near.copy(); any_obj = np.zeros((H, W), bool)
        for o in tracks.objects:
            m = o.masks.get(fi)
            if m is not None: mm = _resize_mask(m, (W, H)); static_mask &= ~mm; any_obj |= mm
        if geom.dynamic_mask is not None: static_mask &= ~geom.dynamic_mask[k].astype(bool)
        if geom.depth_conf is not None:
            conf = geom.depth_conf[k]; thr = np.percentile(conf, 30)
            if conf.max() - conf.min() > 1e-6: static_mask &= conf >= thr   # skip the filter when confidence is constant (ground-truth backend)
        pts_all.append(to_world(unproject(depth, geom.intrinsics[k], static_mask, stride=4), _fix(geom.cam_to_world[k])))
        if any_obj.any(): obj_pts.append(to_world(unproject(depth, geom.intrinsics[k], any_obj & near, stride=6), _fix(geom.cam_to_world[k])))
    P = np.concatenate(pts_all, 0) if pts_all else np.zeros((0, 3))
    O = np.concatenate(obj_pts, 0) if obj_pts else np.zeros((0, 3))
    rng = rng or np.random.default_rng(0)
    if len(P) > 60000: P = P[rng.choice(len(P), 60000, replace=False)]
    info = {"n_static_points": int(len(P)), "n_object_points": int(len(O))}
    up = np.array([0, 1.0, 0]); cam0 = _fix(geom.cam_to_world[0])[:3, 3]
    scene_scale = float(np.percentile(np.abs(P - P.mean(0)).max(1), 50)) if len(P) else 1.0
    thresh = max(scene_scale * pc["ground_ransac_thresh_m"], 1e-4)
    T = np.eye(4)
    # iterative plane extraction
    cands, rest = [], P.copy()
    for _ in range(4):
        if len(rest) < 200: break
        n, d, inl = fit_plane_ransac(rest, thresh=thresh, rng=rng)
        if n is None or inl.sum() < 100: break
        if n @ up < 0: n, d = -n, -d
        ang = math.degrees(math.acos(np.clip(n @ up, -1, 1)))
        cam_h = float(cam0 @ n + d)                         # signed height of camera above plane
        frac_obj_above = float(((O @ n + d) > -thresh).mean()) if len(O) else 1.0
        cands.append({"idx": len(cands), "n": n, "d": d, "inliers": int(inl.sum()), "angle_deg": round(ang, 1), "cam_height": round(cam_h, 4), "frac_obj_above": round(frac_obj_above, 3)})
        rest = rest[~inl]
    fwd = (_fix(geom.cam_to_world[0])[:3, :3] @ np.array([0, 0, 1.0]))  # camera viewing direction in the y-up world (OpenCV +z)
    for c in cands: c["angle_to_view_deg"] = round(math.degrees(math.acos(np.clip(abs(c["n"] @ fwd), -1, 1))), 1)
    ok = [c for c in cands if c["angle_deg"] < 50 and c["cam_height"] > 0 and (not len(O) or c["frac_obj_above"] >= 0.5)]
    if not ok:  # top-down / steep views: the dominant plane faces the camera
        ok = [c for c in cands if c["angle_to_view_deg"] < 35 and (not len(O) or c["frac_obj_above"] >= 0.3)]
        for c in ok:  # orient the normal towards the camera so the camera ends up above the ground
            if c["cam_height"] < 0: c["n"], c["d"], c["cam_height"] = -c["n"], -c["d"], -c["cam_height"]
        if ok: info["ground_mode"] = "facing_camera"
    info["plane_candidates"] = [{k: v for k, v in c.items() if k not in ("n", "d")} for c in cands]
    if not ok:
        info["ground"] = "not_found"; return T, 1.0, info
    best = max(ok, key=lambda c: c["inliers"] * (0.5 + 0.5 * c["frac_obj_above"]))
    normal, d = best["n"], best["d"]
    info["ground"] = "candidate_%d" % best["idx"]; info["ground_normal_angle_to_up_deg"] = best["angle_deg"]; info["ground_inliers"] = best["inliers"]
    Rg = rotation_aligning(normal, up)
    T[:3, :3] = Rg
    p0 = -d * normal; T[:3, 3] = np.array([0, -(Rg @ p0)[1], 0])
    cam0h = (T @ _fix(geom.cam_to_world[0]))[:3, 3]
    h = float(cam0h[1]); target = pc.get("assumed_camera_height_m", 1.2)
    scale = target / h if geom.scale == "relative" and h > 1e-3 else 1.0
    info["camera0_height_before_scale"] = round(h, 4); info["scale_factor"] = round(scale, 4)
    return T, scale, info

def _fix(c2w: np.ndarray) -> np.ndarray:
    """Convert a cam->world matrix in OpenCV-world convention to a y-up world (flip world y,z)."""
    F = np.eye(4); F[:3, :3] = CV2THREE
    return F @ c2w

from .masks import resize_mask as _resize_mask

# ---------------- OBB per frame ----------------
def obb_from_points(P: np.ndarray) -> dict | None:
    if len(P) < 20: return None
    # robust trim
    med = np.median(P, 0); dev = np.linalg.norm(P - med, axis=1); keep = dev < np.percentile(dev, 90) * 1.5 + 1e-6
    P = P[keep]
    if len(P) < 20: return None
    xz = P[:, [0, 2]]; c = xz.mean(0); cov = np.cov((xz - c).T)
    w, v = np.linalg.eigh(cov); ax = v[:, 1]; yaw = math.atan2(ax[1], ax[0])  # angle of principal axis in xz
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, sy], [-sy, cy]]); loc = (xz - c) @ Rz.T
    lo = np.percentile(loc, 3, axis=0); hi = np.percentile(loc, 97, axis=0)
    ylo, yhi = np.percentile(P[:, 1], 3), np.percentile(P[:, 1], 97)
    center_loc = (lo + hi) / 2; center_xz = c + center_loc @ Rz
    size = np.array([max(hi[0] - lo[0], 0.02), max(yhi - ylo, 0.02), max(hi[1] - lo[1], 0.02)])
    quat = R.from_euler("y", -yaw).as_quat()  # rotation about y that maps local x axis to principal axis
    return {"center": [float(center_xz[0]), float((ylo + yhi) / 2), float(center_xz[1])], "quat": [float(q) for q in quat], "size": [float(s) for s in size], "n": int(len(P))}

def continuous_footprint_yaw(quats: np.ndarray, aspect_ratio: float,
                             square_below: float = 1.6) -> np.ndarray:
    """把足迹主轴的朝向序列接成连续的，再交给运动估计。

    `obb_from_points` 的 yaw 来自 xz 点云 PCA 的主轴，它带两重**规范自由度**：

      特征向量的符号（180 度）—— 在任何形状上都是任意的，`eigh` 给哪个方向没有意义；
      长短轴互换（90 度）—— 足迹接近正方形时两个特征值几乎相等，谁是主轴也是任意的。

    两者都会在帧之间跳，而下游把跳变直接读成转动。实测（真值感知、真值静止的物体）：
    底面长短边比 1.2–1.5 的那一档，yaw 的 5–95 分位跨度中位 **73.5 度**；
    比值 2.5 以上的只有 5.1 度；log(比值) 与 log(跨度) 的相关系数 **−0.906**。
    #63 里一个真值**静止**的物体被报出 76.5 度转角、一个**纯平移**的被报出 38.1 度，
    都是这个。

    做法是逐帧在等价的候选里挑离上一帧最近的那个——消掉的是规范自由度，不是信号。
    180 度一直允许。90 度只在足迹比低于 `square_below` 时允许：细长物体的直角转动
    是真的，允许 90 度会把它抹掉。

    一条原理上的限制：足迹主轴的朝向**只定义到 180 度**，长方形转过半圈和没转是
    同一个足迹。所以转超过半圈的 spin 类物体，靠足迹 yaw 读不出来，这不是实现问题，
    要换信号（点云本身的朝向，或者纹理）才能解决。
    """
    q = np.atleast_2d(np.asarray(quats, float))
    if len(q) < 2:
        return np.asarray(quats, float)
    # scipy 1.17 起，单轴序列要求角度数组的最后一维等于轴数，所以这里一律用 (N, 1)
    ang = R.from_quat(q).as_euler("yxz")[:, 0]
    step = (np.pi / 2) if aspect_ratio < square_below else np.pi
    out = [float(ang[0])]
    for a in ang[1:]:
        k = round((out[-1] - a) / step)
        out.append(float(a + k * step))
    return R.from_euler("y", np.asarray(out)[:, None]).as_quat()


def smooth_series(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) < 3 or win <= 1: return x
    win = min(win | 1, len(x) if len(x) % 2 else len(x) - 1); k = np.ones(win) / win
    pad = win // 2; xp = np.pad(x, ((pad, pad), (0, 0)) if x.ndim == 2 else (pad, pad), mode="edge")
    return np.stack([np.convolve(xp[:, j], k, mode="valid") for j in range(x.shape[1])], 1) if x.ndim == 2 else np.convolve(xp, k, mode="valid")

# ---------------- main ----------------
def build_evidence(clip: str, frames: list[dict], geom: Geometry, tracks: Tracks, cfg: dict, out_dir: str | Path, keyframes: list[dict], fallbacks: list[str], phrases_source: str = "") -> dict:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    W, H = geom.size()
    T, scale, align_info = align_world(geom, tracks, cfg)
    def wpt(pts_cam, k):  # camera points -> aligned, scaled world
        return (to_world(pts_cam, _fix(geom.cam_to_world[k])) @ T[:3, :3].T + T[:3, 3]) * scale
    cam_poses = []
    for k, fi in enumerate(geom.frame_indices):
        M = T @ _fix(geom.cam_to_world[k]); pos = M[:3, 3] * scale
        # three.js camera looks down -Z: convert OpenCV cam axes (x r, y down, z fwd) -> (x r, y up, z back)
        Rw = M[:3, :3] @ CV2THREE
        cam_poses.append({"t": frames[fi]["t"], "pos": [float(v) for v in pos], "quat": _quat_xyzw(Rw), "conf": 1.0})
    K0 = geom.intrinsics[0]; fov_deg = float(2 * math.degrees(math.atan(H / (2 * K0[1, 1]))))
    # static extent from ground points
    gpts = []
    for k in range(0, len(geom.frame_indices), max(1, len(geom.frame_indices) // 6)):
        fi = geom.frame_indices[k]; m = np.ones((H, W), bool)
        for o in tracks.objects:
            mm = o.masks.get(fi)
            if mm is not None: m &= ~_resize_mask(mm, (W, H))
        pts = wpt(unproject(geom.depth[k], geom.intrinsics[k], m, stride=6), k)
        gpts.append(pts[np.abs(pts[:, 1]) < 0.15 * max(scale, 1e-6) * 3])
    G = np.concatenate(gpts, 0) if gpts else np.zeros((0, 3))
    if len(G) > 100:
        lo, hi = np.percentile(G, 2, axis=0), np.percentile(G, 98, axis=0)
        ground = {"kind": "ground", "normal": [0, 1, 0], "offset": 0.0, "coverage": float(len(G)), "conf": 0.8 if align_info.get("ground", "").startswith("candidate") and align_info.get("ground_inliers", 0) > 500 else 0.5,
                  "center_hint": [float((lo[0] + hi[0]) / 2), 0.0, float((lo[2] + hi[2]) / 2)], "extent_hint": [float(max(hi[0] - lo[0], 1.0)), 0.2, float(max(hi[2] - lo[2], 1.0))]}
    else:
        ground = {"kind": "ground", "normal": [0, 1, 0], "offset": 0.0, "coverage": 0.0, "conf": 0.2, "center_hint": [0, 0, 0], "extent_hint": [10, 0.2, 10]}
    # per object
    use_contact = bool(cfg["perception"].get("ground_contact_centers", True)) and align_info.get("ground") not in (None, "not_found")
    def contact_center(mres, k, size_y):
        """xz of the object's ground-contact point: ray through the lowest mask pixel (median column) intersected with y=0."""
        ys, xs = np.where(mres)
        if len(ys) < 4: return None
        row = int(np.percentile(ys, 98)); cols = xs[ys >= row - 1]
        u, v = float(np.median(cols)), float(row); K = geom.intrinsics[k]
        d_cam = np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])
        M = T @ _fix(geom.cam_to_world[k]); origin = M[:3, 3] * scale; direction = M[:3, :3] @ d_cam
        if direction[1] >= -1e-6: return None
        lam = -origin[1] / direction[1]; hit = origin + lam * direction
        if lam <= 0 or np.linalg.norm(hit - origin) > 60: return None
        return [float(hit[0]), float(size_y / 2), float(hit[2])]
    objects = []
    visibility_provenance = {}
    dropped_tracks: list[dict] = []
    appearances = observe_colors(frames, {o.id: o.masks for o in tracks.objects}, tracks.frame_size,
                                 cfg["perception"].get("color_min_pixels", 4),
                                 cfg["perception"].get("color_aspect_tolerance", 0.02))
    for o in tracks.objects:
        registration = None   # 配准残差，只有走配准那条路的物体才有
        obbs, observations, ts, centers, quaternions, clouds = [], [], [], [], [], []
        best_frame, best_area = None, 0
        is_structure = is_static_class(o.phrase, cfg)
        for k, fi in enumerate(geom.frame_indices):
            m = o.masks.get(fi)
            if m is None: continue
            mres = _resize_mask(m, (W, H)); area = float(mres.mean())
            pts = wpt(unproject(geom.depth[k], geom.intrinsics[k], mres, stride=2), k)
            ob = obb_from_points(pts)
            if ob is None: continue
            if use_contact and not is_structure:
                cc = contact_center(mres, k, ob["size"][1])
                # only trust the ground-contact point when the object really rests on the ground: the depth-based centre must agree in xz
                # and its bottom must be near y=0 (floating objects such as coins keep the depth-based centre)
                if cc is not None:
                    diag = float(np.linalg.norm(ob["size"])); dxz = math.hypot(cc[0] - ob["center"][0], cc[2] - ob["center"][2])
                    bottom = ob["center"][1] - ob["size"][1] / 2
                    if dxz < max(1.0 * diag, 0.2) and abs(bottom) < max(0.5 * ob["size"][1], 0.15):
                        ob["center_depth"] = ob["center"]; ob["center"] = cc
            ob["t"] = frames[fi]["t"]; ob["frame"] = fi; ob["conf"] = float(min(1.0, ob["n"] / 400)); ob["visible"] = True
            ys, xs = np.where(mres)
            bbox = ({"format": "xyxy", "values": [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                     "space": "pixel", "image_size": [int(W), int(H)]} if len(xs) else "unknown")
            valid_depth = geom.depth[k][mres & (geom.depth[k] > 1e-4)]
            depth = ({"min": float(valid_depth.min()), "median": float(np.median(valid_depth)), "max": float(valid_depth.max()),
                      "unit": "m" if geom.scale == "metric" else "relative"} if len(valid_depth) else "unknown")
            measurement = o.visibility_measurements.get(fi)
            visible_fraction, visibility_reason = checked_visibility(
                measurement, o.id, int(fi), float(ob["t"]), (W, H), geom.source_program_sha256,
                cfg["perception"].get("visibility_time_tolerance_s", 1e-4))
            visibility_provenance.setdefault(o.id, []).append({
                "frame_index": int(fi), "fraction": visible_fraction, "reason": visibility_reason,
                "reference": copy.deepcopy(measurement) if measurement else "unknown"})
            observations.append({"frame_index": int(fi), "track_id": o.id, "t": float(ob["t"]), "bbox": bbox,
                                 "mask_ref": f"masks.npz#{o.id}__{fi}", "visible_fraction": visible_fraction, "depth": depth,
                                 "source": {"geometry": geom.backend or "unknown", "segmentation": tracks.backend or "unknown", "tracking": "mask_centroid_obb"},
                                 "confidence": {"geometry": ob["conf"], "segmentation": float(min(1.0, o.score)), "tracking": "unknown"}})
            obbs.append(ob); ts.append(ob["t"]); centers.append(ob["center"]); quaternions.append(ob["quat"])
            clouds.append(pts)
            if area > best_area: best_area, best_frame = area, fi
        if not obbs: continue
        # 只在很少几帧里被看到的轨迹基本是噪声：实测被看到不到 10% 帧的物体，
        # 运动估计成功率是 0%，而它们占了预测物体的一大块，把精确率压得很低
        # （五十条小批量上真值 5 个物体、预测 11.5 个）。这里按被看到的帧数比例过滤。
        min_frac = float(cfg["perception"].get("min_track_frame_frac", 0.0))
        if min_frac > 0 and len(geom.frame_indices) and len(obbs) / len(geom.frame_indices) < min_frac:
            dropped_tracks.append({"id": o.id, "frames": len(obbs),
                                   "frac": round(len(obbs) / len(geom.frame_indices), 3)})
            continue
        ts_a, c_a, q_a = np.asarray(ts), np.asarray(centers), np.asarray(quaternions)
        # smooth centers/sizes in the stored OBBs too
        cs = smooth_series(c_a, cfg["perception"]["motion"]["smooth_window"]); sz = smooth_series(np.asarray([ob["size"] for ob in obbs]), 9)
        for i, ob in enumerate(obbs): ob["center"] = [float(v) for v in cs[i]]; ob["size"] = [float(v) for v in sz[i]]
        median_size = np.median(sz, axis=0)
        footprint_ratio = max(median_size[0], median_size[2]) / max(min(median_size[0], median_size[2]), 1e-6)
        # 朝向序列先消掉主轴的规范自由度（符号和长短轴互换），再往下传。不做这一步，
        # 跳变会被运动估计读成转动。阈值可配，默认 1.6 —— 实测足迹比 1.5–2.5 这一档
        # 噪声中位仍有 34 度，但真会转的物体也挤在这个带里，所以 90 度那条只在更方的
        # 形状上放开（见 #7 里「抬门槛会挡掉十四个真转动里的九个」那张表）。
        q_a = continuous_footprint_yaw(
            q_a, footprint_ratio,
            float(cfg["perception"]["motion"].get("yaw_square_below", 1.6)))
        for i, ob in enumerate(obbs): ob["quat"] = [float(v) for v in q_a[i]]
        # 喂给运动估计的位置换成跨帧配准的结果。包围盒中心是可见点在各轴上的分位中点，
        # 可见面一变它就跟着动；配准只解帧与帧之间的变换，不依赖单帧算出一个稳定的量。
        # 实测相对第一帧的位移误差 0.253 -> 0.188 m（见 scripts/probe_motion_signal.py）。
        # 留一个开关是因为静止物体的漂移目前还比包围盒中心大，出问题时要能退回去。
        motion_positions, motion_quaternions = c_a, q_a
        motion_source, articulation = "obb_centre", None
        if cfg["perception"]["motion"].get("use_registration", True) and len(clouds) >= 6:
            tracked = track_rigid(clouds)
            # 配准残差按物体记下来。它和 motion_guess.residual 是两回事：那个是运动
            # 假设拟合轨迹的残差，这个是「这团点在帧间能不能当成一个刚体对齐」。
            # 后者不依赖任何运动模型，是独立的一维信息——#56 里排除掉的那八个判别子
            # 都没碰到它。按物体自身尺度归一化，否则大物体天然残差大。
            _res = [r for r in (tracked.get("residuals") or []) if r is not None]
            if _res:
                _diag = float(np.linalg.norm(np.median([ob["size"] for ob in obbs], axis=0)))
                registration = {
                    "rmse_m": round(float(np.median(_res)), 5),
                    "rmse_over_diagonal": round(float(np.median(_res)) / max(_diag, 1e-9), 4),
                    "frames": len(clouds),
                }
            if tracked["ok"] and len(tracked["positions"]) == len(c_a):
                diag = float(np.linalg.norm(np.median([ob["size"] for ob in obbs], axis=0)))
                def _registered_quats(base_quat):
                    """把配准解出的朝向接到第一帧的包围盒朝向上。

                    返回的是绝对朝向序列，和 q_a 同一个约定，所以可以直接替换。
                    """
                    base = R.from_quat(base_quat)
                    return np.asarray([(R.from_matrix(np.asarray(m, float)) * base).as_quat()
                                       for m in tracked["orientations"]], float)

                # 静止判据默认关着。实测它在真值静止的物体上**一次都没触发过**，
                # 却在 5.4% 的运动物体上误触发。根因是每步位移这个量分不开两类：
                # 静止物体每步位移是对角线的 0.165，运动物体是 0.179，两条分布重合。
                # 不是阈值没调好，是判据本身没有区分力。
                if cfg["perception"]["motion"].get("use_static_detector", False) \
                        and looks_static(tracked, diag):
                    # 每步修正都接近单位阵，说明物体没动。这种情况下配准的累积轨迹会被
                    # 缓慢滑移带偏（实测静止漂移 0.286 m，比包围盒中心的 0.182 m 还大），
                    # 所以把位置钉在第一帧，让静止这个假设干干净净地成立。
                    motion_positions = np.repeat(c_a[:1], len(c_a), axis=0)
                    # 位置钉住了，朝向也要钉住。只钉位置的话，包围盒朝向的抖动会和
                    # 「静止」这个假设打架，静止候选反而拿不到高分。
                    motion_quaternions = np.repeat(q_a[:1], len(q_a), axis=0)
                    motion_source = "registration_static"
                else:
                    registered = np.asarray(tracked["positions"], float)
                    # 配准给的是相对锚点的轨迹，锚点落在可见表面上。平移到第一帧的包围盒
                    # 中心，保住绝对位置，用它的形状。
                    # 配准的每步噪声是物体尺寸的 17%，而真实运动是平滑的。包围盒中心
                    # 那条路本来就过一道时间平滑，配准这条之前漏了。实测加上之后：
                    # 轨迹误差 0.173 -> 0.140，静止物体漂移 0.231 -> 0.084——后者原本是
                    # 配准相对包围盒中心唯一的退步，平滑之后反超（包围盒中心是 0.170）。
                    window = int(cfg["perception"]["motion"].get("registration_smooth_window", 15))
                    registered = smooth_series(registered, window)
                    motion_positions = registered - registered[0] + c_a[0]
                    if len(tracked.get("orientations") or []) == len(q_a):
                        motion_quaternions = _registered_quats(q_a[0])
                    motion_source = "cross_frame_registration"
                    # 关节推断默认关着。在四十条有标准答案的合成片段上扫过阈值：
                    # 最好的一档「可用率」（报出来的东西里既是真关节、类型也判对的比例）
                    # 只有 0.333，再收紧就一个都报不出来。报一个假关节比不报更糟——
                    # 用户拖一下就会发现。等运动信号再好一些再打开。
                    if cfg["perception"]["motion"].get("infer_articulation", False):
                        articulation = infer_articulation(tracked)
        motion_result = estimate_motion(ts_a, motion_positions, motion_quaternions, cfg, size=median_size, coordinate_space="world",
                                        visible_fractions=np.asarray([v["visible_fraction"] for v in observations])
                                        if all(isinstance(v["visible_fraction"], (int, float)) for v in observations) else None,
                                        camera_pose_available=geom.backend != "static",
                                        orientation_reliable=footprint_ratio >= cfg["perception"]["motion"].get("yaw_min_aspect_ratio", 1.3))
        mg = motion_result["motion_guess"]
        mg["position_source"] = motion_source
        # contacts with ground
        bottoms = cs[:, 1] - sz[:, 1] / 2
        on_ground = np.abs(bottoms) < cfg["perception"]["contact_dist_m"] * max(1.0, float(np.median(sz[:, 1])) / 0.1)
        contacts = []
        if on_ground.mean() > 0.5: contacts.append({"with_id": "ground", "t_start": float(ts_a[0]), "t_end": float(ts_a[-1]), "conf": float(on_ground.mean())})
        geometry_conf = float(np.mean([ob["conf"] for ob in obbs])) if obbs else "unknown"
        objects.append({"id": o.id, "track_id": o.id, "class_guess": o.phrase, "confidence": float(min(1.0, o.score)), "is_dynamic": "unknown" if mg["type"] == "unknown" else mg["type"] != "static",
                        "obb": [{k: v for k, v in ob.items() if k != "n"} for ob in obbs], "motion_guess": mg, "contacts": contacts,
                        **({"registration": registration} if registration else {}),
                        "observations": observations,
                        "appearance": appearances[o.id],
                        **({"articulation": articulation} if articulation else {}),
                        "attribute_confidence": {"class": float(min(1.0, o.score)), "identity": "unknown", "geometry": geometry_conf,
                                                 "motion": float(mg["conf"]) if isinstance(mg.get("conf"), (int, float)) else "unknown"},
                        "hypotheses": copy.deepcopy(o.identity_hypotheses) + ([motion_result["hypothesis"]] if motion_result["hypothesis"] else []),
                        "best_frame": best_frame, "mask_area_frac": float(best_area), "notes": f"{len(obbs)} frames with 3D points"})
    association_diagnostics = copy.deepcopy(tracks.association_diagnostics)
    retained_ids = {obj["id"] for obj in objects}
    for obj in objects:
        for hypothesis in obj["hypotheses"]:
            for candidate in hypothesis.get("candidates", []):
                ref = candidate.get("ref")
                if ref and ref not in retained_ids:
                    candidate.pop("ref")
                    association_diagnostics.append({"code": "unresolved_identity_candidate", "object_id": obj["id"], "candidate": ref})
    evidence = {
        "schema_version": "2.0",
        "meta": {"clip": clip, "fps_sampled": float(round(1 / max(frames[1]["t"] - frames[0]["t"], 1e-6), 3)) if len(frames) > 1 else 0.0, "n_frames": len(frames), "duration": float(frames[-1]["t"]),
                 "geometry_backend": geom.backend, "segmentation_backend": tracks.backend, "tracks_backend": "mask_centroid_obb", "fallbacks": fallbacks,
                 "scale": geom.scale, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "alignment": align_info, "phrases_source": phrases_source, "geometry_frames": geom.frame_indices,
                 "association_diagnostics": association_diagnostics,
                 "dropped_short_tracks": dropped_tracks,
                 "visibility_measurements": visibility_provenance},
        "frames": [{"frame_index": int(frame["index"]), "t": float(frame["t"]),
                    "source": {"kind": "video_frame", "ref": Path(frame["file"]).name if frame.get("file") else "unknown"}} for frame in frames],
        "camera": {"intrinsics": {"fx": float(K0[0, 0]), "fy": float(K0[1, 1]), "cx": float(K0[0, 2]), "cy": float(K0[1, 2]), "width": int(W), "height": int(H), "fov_deg": fov_deg}, "poses": cam_poses,
                   "note": "poses in aligned world (y up, ground y=0); quaternion is three.js camera orientation (looks down -Z)"},
        "static": {"planes": [ground], "bounds": {"ground_center": ground["center_hint"], "ground_extent": ground["extent_hint"]}},
        "objects": objects,
        "keyframes": keyframes,
    }
    (out / "evidence.json").write_text(json.dumps(evidence, indent=1))
    try:
        from .overlay import render_overlay
        render_overlay(frames, geom, tracks, evidence, T, scale, out / "overlay")
    except Exception as e:  # 叠加视频只是给人看的，画不出来不影响证据
        (out / "overlay_error.txt").write_text(repr(e))
    return evidence

