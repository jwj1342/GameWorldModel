"""读 GLB 里的顶点，不依赖 trimesh / pygltflib。

为什么自己写：集群的 venv 里没有任何 glTF 库，而我们有两处需要资产的真实几何——

  #7：把**完整的**资产网格当配准模板。现在是让各帧的局部点云互相配，而局部点云
      随视角扫动，ICP 把「可见面换了」解成「物体转了」（真值静止的物体被解出
      19.85 度）。有一边是钉死的完整形状就能断掉这一环。
  #64：同类尺寸离谱的判据现在拿「同一片段里同类其它实例的尺寸中位数」当参照，
      实例少的时候中位数会被错的那个带着走。资产的标称尺寸是固定的，更稳。

Kenney 的资产是没有压缩的 glTF 二进制，所以只需要处理最常见的那条路径：
单个 BIN 块、float32 的 POSITION、uint16/uint32 的索引。碰到别的（Draco 压缩、
稀疏访问器）就老实返回 None，不猜。
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

_FLOAT32 = 5126
_COMPONENT = {5120: np.int8, 5121: np.uint8, 5122: np.int16,
              5123: np.uint16, 5125: np.uint32, _FLOAT32: np.float32}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def _chunks(data: bytes):
    if len(data) < 12 or data[:4] != b"glTF":
        return None, None
    offset, js, bin_ = 12, None, None
    while offset + 8 <= len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        body = data[offset + 8: offset + 8 + length]
        if kind == 0x4E4F534A:
            js = json.loads(body.decode("utf-8"))
        elif kind == 0x004E4942:
            bin_ = body
        offset += 8 + length + (-length % 4)
    return js, bin_


def _accessor(gltf: dict, buf: bytes, index: int) -> np.ndarray | None:
    acc = gltf["accessors"][index]
    if "sparse" in acc or acc.get("bufferView") is None:
        return None                                   # 稀疏访问器不处理，不猜
    dtype = _COMPONENT.get(acc["componentType"])
    per = _COUNT.get(acc["type"])
    if dtype is None or per is None:
        return None
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride")
    item = np.dtype(dtype).itemsize * per
    if stride and stride != item:                     # 交错存放：逐条取
        rows = [np.frombuffer(buf, dtype=dtype, count=per,
                              offset=start + i * stride) for i in range(acc["count"])]
        return np.asarray(rows, float)
    flat = np.frombuffer(buf, dtype=dtype, count=acc["count"] * per, offset=start)
    return flat.reshape(acc["count"], per).astype(float)


def _node_matrix(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], float).reshape(4, 4).T   # glTF 是列优先
    m = np.eye(4)
    if "rotation" in node:
        from scipy.spatial.transform import Rotation as R
        m[:3, :3] = R.from_quat(node["rotation"]).as_matrix()
    if "scale" in node:
        m[:3, :3] = m[:3, :3] @ np.diag(np.asarray(node["scale"], float))
    if "translation" in node:
        m[:3, 3] = np.asarray(node["translation"], float)
    return m


def load_extent(path: str | Path) -> np.ndarray | None:
    """Static GLB XYZ bounds, using the same local-box transform as Three Box3.

    Keep axis order and every active node instance. Surface sampling and sorted
    ratios cannot determine the rendered height. Unsupported/deformed geometry
    returns None instead of a partial, underestimated bound.
    """
    try:
        gltf, buf = _chunks(Path(path).read_bytes())
        if gltf is None or buf is None or not gltf.get("scenes"):
            return None
        points = []

        def walk(index: int, parent: np.ndarray, ancestors: frozenset) -> None:
            if index in ancestors:
                raise ValueError("cyclic GLB node")
            node = gltf["nodes"][index]
            matrix = parent @ _node_matrix(node)
            if "skin" in node or node.get("weights"):
                raise ValueError("deformed GLB is unsupported")
            if "mesh" in node:
                mesh = gltf["meshes"][node["mesh"]]
                if mesh.get("weights"):
                    raise ValueError("morph weights are unsupported")
                for primitive in mesh["primitives"]:
                    if primitive.get("targets") or primitive.get("extensions", {}).get("KHR_draco_mesh_compression"):
                        raise ValueError("unsupported primitive")
                    accessor = primitive["attributes"]["POSITION"]
                    spec = gltf["accessors"][accessor]
                    if spec.get("componentType") != _FLOAT32 or spec.get("normalized"):
                        raise ValueError("quantized positions are unsupported")
                    position = _accessor(gltf, buf, accessor)
                    if position is None or position.ndim != 2 or position.shape[1] != 3 or not len(position):
                        raise ValueError("missing positions")
                    if not np.isfinite(position).all():
                        raise ValueError("nonfinite positions")
                    lo, hi = position.min(0), position.max(0)
                    # Box3.setFromObject transforms the geometry's local AABB.
                    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                    points.append(corners @ matrix[:3, :3].T + matrix[:3, 3])
            for child in node.get("children", []):
                walk(child, matrix, ancestors | {index})

        scene = gltf["scenes"][gltf.get("scene", 0)]
        for index in scene.get("nodes", []):
            walk(index, np.eye(4), frozenset())
        if not points:
            return None
        extent = np.ptp(np.vstack(points), axis=0)
        return extent if np.isfinite(extent).all() and (extent > 1e-9).all() else None
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return None


def load_points(path: str | Path, samples: int = 4000, seed: int = 0) -> np.ndarray | None:
    """在资产表面上采点，返回 (N, 3)。读不动就返回 None。

    按三角形面积加权采样，而不是直接用顶点——低多边形资产的顶点太稀，
    大面上一个点都没有，当配准模板不够用。
    """
    data = Path(path).read_bytes()
    gltf, buf = _chunks(data)
    if gltf is None or buf is None or not gltf.get("meshes"):
        return None
    if any("KHR_draco_mesh_compression" in (p.get("extensions") or {})
           for m in gltf["meshes"] for p in m.get("primitives", [])):
        return None                                   # 压缩过的不处理

    world: dict[int, np.ndarray] = {}

    def walk(index: int, parent: np.ndarray) -> None:
        node = gltf["nodes"][index]
        here = parent @ _node_matrix(node)
        if node.get("mesh") is not None:
            world.setdefault(node["mesh"], here)
        for child in node.get("children", []):
            walk(child, here)

    scene = gltf.get("scenes", [{}])[gltf.get("scene", 0)]
    for root in scene.get("nodes", range(len(gltf.get("nodes", [])))):
        walk(root, np.eye(4))

    tris = []
    for mesh_index, mesh in enumerate(gltf["meshes"]):
        matrix = world.get(mesh_index, np.eye(4))
        for prim in mesh.get("primitives", []):
            pos_index = (prim.get("attributes") or {}).get("POSITION")
            if pos_index is None:
                continue
            pos = _accessor(gltf, buf, pos_index)
            if pos is None:
                continue
            pos = pos @ matrix[:3, :3].T + matrix[:3, 3]
            idx = (_accessor(gltf, buf, prim["indices"]).astype(int).ravel()
                   if prim.get("indices") is not None else np.arange(len(pos)))
            if len(idx) < 3:
                continue
            tris.append(pos[idx[: len(idx) // 3 * 3]].reshape(-1, 3, 3))
    if not tris:
        return None
    tri = np.vstack(tris)
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        return None
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(tri), size=samples, p=area / total)
    u, v = rng.random((samples, 1)), rng.random((samples, 1))
    flip = (u + v > 1).ravel()
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    return a[pick] + u * (b[pick] - a[pick]) + v * (c[pick] - a[pick])
