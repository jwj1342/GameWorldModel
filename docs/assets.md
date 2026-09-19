# 素材层

> 视频中检测到的物体如何变成 three.js 里的几何：三层回退、CC0 素材库、检索配方、生成模型、许可。数据来自 2026-09-18 调研（`../survey/03`）。

## 1. 三层回退

| 层 | 做法 | 何时用 |
|---|---|---|
| 参数化 primitive | 门、平台、金币、树、矿车等各写一个参数化生成器，接受 extent 与颜色 | 永远可用；第一阶段到第五阶段的默认 |
| CC0 GLB 库 + 检索 | 1,000~2,000 个低多边形 GLB，CLIP 检索 + VLM 重排 | 第六阶段起的默认 |
| 单图生成 | TRELLIS.2-4B（MIT）或 SAM 3D Objects 对 hero 物体生成 | 可选；每条片段 1~3 个，失败回退 |

不建 Objaverse 级大库：CC0 仅 3.5K，质量噪声大，检索需 Objaverse++ 级过滤，与 baseline 目标不匹配。

## 2. CC0 素材库来源

| 库 | 规模 | 格式 | 许可 | 获取 | 品类 |
|---|---|---|---|---|---|
| **Kenney** | 4,812 个 GLB / 49 套 kit（镜像 github.com/shorepine/kenney，附 kits.tsv） | GLB | CC0 | git clone | Nature 329、Platformer 153（金币、平台）、Car 50、Train 103、Racing 112、City、Castle 76、Pirate 72、Food 200、Furniture 140、Space 153、Characters、Weapon |
| **KayKit** | Dungeon 200+、Adventurers 4 角色（75 动画）、City Builder | FBX / glTF | CC0 1.0 | GitHub org `KayKit-Game-Assets` | 地牢、角色、城市 |
| **Quaternius** | 数千模型，60~70% 免费 | FBX / OBJ / glTF | CC0 | Google Drive 逐包；多数已镜像到 Poly Pizza | 角色（含动画）、载具、自然、建筑、道具 |
| Poly Pizza | 10,700+ | OBJ / FBX / glTF | 混合，API 可按 CC0 过滤 | API v1.1，免费 key；**计算节点代理封禁，只能在登录节点抓** | 全品类低多边形 |
| Poly Haven | 约 520 模型 | glTF | CC0 | 公共 API | 写实 PBR 扫描，非低多边形 |
| Sketchfab | 700K+ CC | glTF | 混合 | 下载需终端用户 OAuth，署名随行 | 不适合打包 |
| Mixamo | 角色与动画 | FBX | 免版税但禁止独立再分发 | 无 API | 不打包；用 KayKit 角色替代 |

Kenney + KayKit + Quaternius 全部 CC0、全部有 glTF，合计超过 5,000 个 GLB，品类正好覆盖平台跳跃与低多边形世界。

## 3. 库构建流程

1. 登录节点抓取（网络操作，轻量）：Kenney 镜像按 kits.tsv 选 Nature、Platformer、Car、Train、City、Castle、Pirate、Food、Furniture、Characters 等约 1,500 个；KayKit 与 Quaternius 免费包；Poly Pizza 按 `license=CC0` 补漏。
2. 归一化：居中、Y-up、单位包围盒、记录原始尺寸、meshopt 或 Draco 压缩；记录来源、kit、许可。
3. 计算节点批量渲染每个资产 4~6 个视图的缩略图（three.js 无头渲染即可，复用 harness）。
4. 嵌入：CLIP 或 SigLIP 图像嵌入 + 用 kit 名、文件名、VLM 描述做文本嵌入，入 FAISS。
5. 存放：GLB 与嵌入放 `/project`（约 0.5~2 GB GLB + 0.2~0.4 GB 缩略图 + 十几 MB 嵌入）。

## 4. 检索配方

Holodeck（CVPR 2024）的配方去掉几何项再加上图像查询：查询 = 视频中该物体的最佳裁剪帧（图像到图像）+ 类别文本（文本到图像与文本）→ 加权融合 → top-k → VLM 重排（尺寸先验：候选原始尺寸与 evidence 中 extent 的比值；是否 rigged；风格一致性）→ 按 extent 缩放放置。命中分低于阈值时回退 primitive。

对不超过两千个资产的小库，渲染缩略图 + CLIP 已足够；OpenShape / Uni3D 等 3D 原生嵌入是可选增强。

## 5. 生成模型

| 模型 | 发布 | 输入 | 输出 | 显存 / 时延 | 许可 | 评价 |
|---|---|---|---|---|---|---|
| **TRELLIS.2-4B**（Microsoft） | 2025-12 | 单图 | mesh + PBR → GLB | ≥ 24 GB；H100 上 512³ 约 3 秒、1024³ 约 17 秒；L40S 估 2~3 倍 | MIT | **首选**：自带 decimate、remesh、UV unwrap，最接近 game-ready |
| **SAM 3D Objects**（Meta） | 2025-11 | 单图 + 掩码 | 形状、纹理、位姿与布局；3DGS 与 mesh | 未核实 | SAM License（HF 需登记） | 擅长遮挡与杂乱场景，最贴近"视频帧裁剪"输入；额外给位姿 |
| Hunyuan3D 2.1 | 2025-06 | 图像 | mesh + PBR | 约 29 GB | Tencent Community License：不适用于 EU / UK / 韩国，禁止用输出训练其他模型 | 质量口碑好但许可受限 |
| Hunyuan3D-Omni | 2025-09 | 图像 + 包围盒 / 骨架等控制 | mesh + PBR | 10 GB | 同族 | 包围盒控制对"按视频尺寸生成"有用 |
| Step1X-3D | 2025-05 | 单图 | 水密 mesh + 纹理 | 27~29 GB；约 150 秒 | Apache-2.0 | 训练与数据管线全开源 |
| TripoSG / Direct3D-S2 / PartCrafter | 2025 | 图像 | mesh（无纹理 / 高分辨 / 多部件） | 8~24 GB | MIT | 几何为主 |
| Hunyuan3D 2.5 / 3.0 | — | — | — | — | 无开源仓库，仅 API | — |

没有开源模型原生输出美术风格低多边形加干净 UV；统一后处理：decimate → xatlas UV → 纹理烘焙 → glTF-Transform 压缩。

商业服务一行速览：Meshy（Pro $20/月，API 在 Pro 以上，Meshy-6 有 low-poly 模式）、Tripo（有公开 API）、Rodin/Hyper3D（Business $120/月含 API）、Sloyd（API 约 $0.13~0.33/模型，参数化 game-ready）。

## 6. 关节与部件

门、抽屉、轮子这类关节物体，baseline 用参数化模板最稳（门 = 框 + 板 + 绕 Y 轴 pivot；矿车 = 箱体 + 四个圆柱轮）。若要从生成 mesh 自动拆关节：Hunyuan3D-Part（P3-SAM）与 Particulate（2025-12，前馈秒级，代码未核实）是最现实的选项；Articulate-Anything（ICLR 2025）能从视频生成 URDF，依赖 PartNet-Mobility 检索。2026 年的 PAct、ArtLLM、URDF-Anything+、MonoArt 均为单图或 mesh 到部件与运动参数，代码状态未核实。

## 7. 视频到规范 mesh

截至 2026-09 没有开箱即用的工具：Shape of Motion、MoSca 输出动态高斯而非 mesh；Mesh4D（2026-01）最贴题但代码未核实。实用做法是选帧（面积最大、遮挡最少）→ SAM 3 掩码 → SAM 3D Objects 或 TRELLIS.2 单图生成。

## 8. 许可规则（面向论文发布）

- 只打包 CC0（Kenney、KayKit、Quaternius、Poly Haven、Smithsonian），可随代码整体分发。
- 引入 CC-BY 必须附 attribution 清单；排除 NC / ND / SA。
- 不打包 Mixamo、Sketchfab 下载物、ShapeNet、Toys4K、3D-FUTURE，改为提供获取脚本。
- 若生成资产将用于训练数据，选 MIT / Apache 系生成模型（TRELLIS.2、TripoSG、Direct3D-S2、PartCrafter、Step1X-3D），避开 Hunyuan3D 系的地域与训练限制。
