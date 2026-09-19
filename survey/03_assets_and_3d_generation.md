# 调研 03：3D 资产策略：CC0 库、研究数据集、text/image-to-3D、关节生成（2026-09-18）

> 由调研 agent 生成；凡未能从一手页面确认的数字/状态均标注 **未核实**。

## 0. 一句话结论
基线应做「三层回退」：primitive 组合（永远可用）→ 约 1,000–2,000 个 CC0 低多边形 GLB 的小型库 + CLIP 检索 → 仅对 hero 物体调用开源 image-to-3D（首选 MIT 许可的 TRELLIS.2）。不需要 Objaverse 级别的大库；存储 < 3 GB，工程量约 1 人周。

## 1. 免费 / CC0 低多边形资产库

| 库 | 规模 | 格式 | 许可 | 程序化获取 | 游戏品类覆盖 |
|---|---|---|---|---|---|
| **Kenney** kenney.nl | 4,812 个 GLB / 49 套 kit（镜像 github.com/shorepine/kenney 附 kits.tsv） | GLB(+glTF/FBX/OBJ) | CC0 | 镜像仓库可 git clone | 最全：Nature 329、Platformer 153（金币/平台）、Car 50、Train 103、Racing 112、City Kit×4、Castle 76、Pirate 72、Food 200、Furniture 140、Space 153、Characters 18+26、Weapon 37 |
| **Quaternius** | 数千模型 / 80+ 包；60–70% 免费 | FBX/OBJ/glTF | CC0 | Google Drive 逐包；大部分镜像到 Poly Pizza | 角色（含动画）、载具、自然、建筑、道具、Platformer kit |
| **KayKit** | Dungeon 200+；Adventurers 4 角色（75 动画）；City Builder | FBX/glTF | CC0 1.0 | GitHub org `KayKit-Game-Assets` 可 clone | 地牢、角色、城市 |
| **Poly Pizza** | 10,700+（含 Kenney/Quaternius/Google Poly 存档） | OBJ/FBX/glTF | 混合（API 可按 CC0 过滤） | **API v1.1**（免费 key）；已有 MCP 客户端 | 全品类低多边形，最适合按 label 检索 |
| Poly Haven | ~520 模型 | glTF/FBX/blend | CC0 | 无 key 公共 API | 写实 PBR 扫描，非低多边形 |
| Sketchfab | 700K+ CC 模型 | glTF/GLB | CC0/BY/NC/ND/SA | Download API 要求终端用户 OAuth；署名随行 | 不适合打包分发 |
| Smithsonian 3D | 2,000+ | OBJ/glTF | CC0 | 网页 | 文物，非游戏道具 |
| OpenGameArt / itch.io CC0 | 4,938 条 / 595 包 | 混杂 | 混合 / CC0 | 无 API | 需清洗 |
| Mixamo (Adobe) | 角色 + 动画 | FBX/DAE | 免版税商用，**禁止独立再分发**；需账号 | 无 API | 不可随论文代码打包 |
| Ready Player Me | — | GLB | 第三方称 2026-01 关闭 **未核实** | 已停 | 不再可用 |

要点：Kenney + KayKit + Quaternius 全部 CC0、全部有 glTF，合计超过 5,000 个 GLB，品类正好是门、矿车、金币、树、平台这类 platformer/低多边形世界。Poly Pizza 是唯一带官方搜索 API 的低多边形站点（注意 Vulcan 计算节点代理封禁 poly.pizza，需在登录节点抓取）。

## 2. 研究级数据集与场景生成系统的检索方式

| 数据集 | 规模 | 许可 | 备注 |
|---|---|---|---|
| Objaverse 1.0 | 798,759 | 逐物体 CC：CC-BY 721K、NC 77K、**CC0 仅 3.5K** | Objaverse-LVIS 子集 47K / 1,156 类 |
| Objaverse-XL | 10.2M | ODC-By；GitHub 部分未声明逐物体许可 | 质量极不均 |
| Objaverse++ | ~500K 精选 | 继承 | 质量四级标注 |
| TRELLIS-500K | 500K | 混合 | 渲染/过滤脚本可复用 |
| ShapeNetCore | 51.3K / 55 类 | 仅非商业研究 | 无纹理为主 |
| ABO | ~7,900 glTF / 63 类 | CC BY 4.0 | 电商产品 |
| Toys4K | ~4K / 105 类 | 不得再分发 | 玩具风 |

场景生成系统的取资产方式：
- **Holodeck**（CVPR 2024）：Objaverse 精选 51K；检索分 = CLIP ViT-L/14 图文相似度（3 视图）+ SBERT 文本相似度 + 包围盒几何相似度；最成熟可照抄的配方。
- **Holodeck 2.0**（2025-08）：放弃检索改为生成，本地 Hunyuan3D 2.1 生成 GLB（A6000 3–5 分钟/件）。
- **SceneCraft**：CLIP 检索器从 Objaverse 取；GPT-4V 看渲染图迭代。
- **Scenix**（2026-08）：不检索而是生成：mask + 描述合成干净单物体参考图 → image-conditioned Hunyuan3D → 按程序 extent 拟合。
- 3D 原生嵌入：OpenShape（Apache-2.0）、Uni3D（MIT）；对 ≤2K 资产的小库，渲染缩略图 + CLIP 足够。

**推荐配方（小库）**：每资产渲染 4–6 视图 → CLIP/SigLIP 图像嵌入 + kit 名/文件名/VLM 描述文本嵌入 → FAISS；查询 = 视频中检测框裁剪图 + label 文本，加权融合 → top-k → VLM 重排（含尺寸先验、是否 rigged）。

## 3. 开源 text/image-to-3D（截至 2026-09）

| 模型 | 发布 | 输入 | 输出 | VRAM / 时延 | 许可 | 游戏就绪性 |
|---|---|---|---|---|---|---|
| TRELLIS (Microsoft) | 2024-12 | 图像/文本 | RF / 3DGS / mesh | ≥16 GB | MIT | 高面数 |
| **TRELLIS.2-4B** | 2025-12 (arXiv 2512.14692) | 单图 | mesh + PBR → GLB | ≥24 GB；H100 512³≈3 s、1024³≈17 s | **MIT** | 自带 decimation / remesh / UV unwrap，最接近 game-ready |
| Hunyuan3D 2.0 / 2.1 | 2025-01 / 2025-06 | 图像 | mesh + PBR | 2.1 合计 29 GB | Tencent Community License：**不适用于 EU/UK/韩国，禁止用输出训练其他模型** | 口碑最好但许可受限 |
| Hunyuan3D 2.5 / 3.0 | — | — | — | — | GitHub 中**不存在**开源仓库（仅 API/Studio） | — |
| Hunyuan3D-Omni | 2025-09 | 图像 + 点云/体素/包围盒/骨架 | mesh + PBR | 10 GB | 同族 | 包围盒控制对"按视频尺寸生成"有用 |
| Hunyuan3D-Part | 2025-09 | mesh | 部件分割 + 部件生成 | 未核实 | 同族 | 门/抽屉部件拆分候选 |
| TripoSG (VAST) | 2025-03 | 图像 | mesh（无纹理） | ≥8 GB | MIT | 可 `--faces 5000` |
| Step1X-3D | 2025-05 | 单图 | 水密 mesh + 纹理 GLB | 27–29 GB；~152 s | Apache 2.0 | 训练与数据管线全开源 |
| Direct3D-S2 | NeurIPS 2025 | 图像 | 高分辨 mesh | 10–24 GB | MIT | 几何为主 |
| **SAM 3D Objects** (Meta) | 2025-11 (arXiv 2511.16624) | 单图 + mask/点/框 | 形状 + 纹理 + **位姿/布局**；3DGS 与 mesh | 未核实 | SAM License（HF gated） | 强于遮挡/杂乱场景，最贴近"视频帧裁剪"输入 |
| PartCrafter | NeurIPS 2025 | 单图 | 多部件 mesh | ≥8 GB | MIT | 基于 TripoSG |
| 2026 新作 | AssetGen（2026-05，单图→受控面数 mesh，30 s）、Pixal3D、MeshWeaver | — | — | — | 代码/权重发布状态均**未核实** | — |

结论：没有开源模型原生输出 artist 风格低多边形 + 干净 UV；TRELLIS.2 的内置 decimate + UV unwrap 最接近。面向论文发布，MIT/Apache 系（TRELLIS.2、TripoSG、Direct3D-S2、PartCrafter、Step1X-3D）更安全。后处理统一走 decimate → xatlas UV → 纹理烘焙 → glTF-Transform 压缩。

商业服务：Meshy Pro $20/月 1,000 credits，API 在 Pro 以上，Meshy-6 有原生 low-poly 模式；Tripo Pro $15.90/月，有公开 API；Rodin/Hyper3D Business $120/月含完整 API；Sloyd API 预付约 $0.13–0.33/模型，参数化 game-ready；Luma Genie / CSM 状态未核实。

## 4. 关节 / 部件感知生成（门、抽屉、轮子）
Articulate-Anything（ICLR 2025，文本/图像/**视频** → VLM 生成 Python → URDF，代码已开）；Real2Code；URDFormer；ArtGS（ICLR 2025）；Articulate AnyMesh（代码 Coming Soon）；Infinite Mobility（过程化 URDF，22 类）；PartCrafter（无关节，MIT）；Hunyuan3D-Part；Particulate（2025-12，前馈 mesh → 部件 + 运动树，秒级，代码未核实）；PAct、ArtLLM、URDF-Anything+、MonoArt（2026 Q1，均为单图/mesh → 部件 + 运动参数，代码状态未核实）；MonoMobility（单目视频零样本运动分析）。

**基线可行路径**：门/矿车轮这类常见关节物体用参数化模板最稳（门 = frame + panel + Y 轴 pivot；矿车 = box + 4 圆柱轮 pivot）；若需从生成 mesh 拆关节，Hunyuan3D-Part 或 Particulate 是最现实的自动化选项。

## 5. 单目视频 → 规范物体 mesh
Shape of Motion（ICCV 2025）、MoSca（CVPR 2025）：动态 3DGS，不输出规范 mesh；GEN3C：不产 mesh；Mesh4D（2026-01，单目物体视频 → 完整 mesh + 变形场，代码未核实）。**结论**：没有开箱即用的视频→规范 mesh 工具；实用做法是选帧（最大、遮挡最少）→ SAM 3 mask → SAM 3D Objects 或 TRELLIS.2 单图生成。

## 6. 基线实践建议
三层回退：
1. **Primitive fallback**（必有）：门/金币/平台/树/矿车各写一个参数化 three.js 生成器，保证 100% 可执行。
2. **小型 CC0 库（1,000–2,000 GLB）+ CLIP 检索**：Kenney（按 kits.tsv 选约 1,500）+ KayKit + Quaternius 免费包；Poly Pizza API 按 `license=CC0` 补漏。统一归一化（居中、单位包围盒、Y-up、Draco/meshopt 压缩）。
3. **Hero 物体生成（可选）**：TRELLIS.2-4B（MIT，24 GB，L40S 48 GB 可单卡跑），或对遮挡帧用 SAM 3D Objects；每视频限 1–3 个物体。

存储估算：2,000 个 GLB ≈ 0.5–2 GB + 缩略图 0.2–0.4 GB + 嵌入 12 MB，合计 < 3 GB。工作量约 1 人周。

许可约束：只打包 CC0（Kenney/KayKit/Quaternius/Poly Haven/Smithsonian）；CC-BY 需附 attribution 清单；排除 NC/ND/SA；不要打包 Mixamo、Sketchfab 下载、ShapeNet/Toys4K/3D-FUTURE。Hunyuan3D 系许可不适用于 EU/UK/韩国且禁止用输出训练其他模型，若要用生成资产做训练数据，选 MIT/Apache 系。

## 主要来源（节选）
Kenney 镜像 https://github.com/shorepine/kenney ；Poly Pizza API https://poly.pizza/docs/api/v1.1 ；KayKit https://github.com/KayKit-Game-Assets ；Objaverse https://objaverse.allenai.org/docs/objaverse-1.0/ ；Objaverse++ https://github.com/TCXX/ObjaversePlusPlus ；Holodeck https://arxiv.org/pdf/2312.09067 ；Holodeck 2.0 https://arxiv.org/html/2508.05899v1 ；Scenix https://arxiv.org/html/2608.07012 ；OpenShape https://github.com/Colin97/OpenShape_code ；Uni3D https://github.com/baaivision/Uni3D ；TRELLIS.2 https://github.com/microsoft/TRELLIS.2 ；Hunyuan3D-2.1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 ；Hunyuan3D-Part https://github.com/Tencent-Hunyuan/Hunyuan3D-Part ；TripoSG https://github.com/VAST-AI-Research/TripoSG ；Step1X-3D https://github.com/stepfun-ai/Step1X-3D ；Direct3D-S2 https://github.com/DreamTechAI/Direct3D-S2 ；PartCrafter https://github.com/wgsxm/PartCrafter ；SAM 3D Objects https://github.com/facebookresearch/sam-3d-objects ；Articulate-Anything https://articulate-anything.github.io/ ；Particulate https://arxiv.org/abs/2512.11798 ；Mesh4D https://arxiv.org/abs/2601.05251 ；Meshy API https://www.meshy.ai/api ；Hyper3D https://hyper3d.ai/pricing ；Sloyd https://www.sloyd.ai/api/pricing
