"""给一个发布包写 README 和署名文件。用法: make_package_docs.py <包目录> <片段名> <标题>"""
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
pkg, clip, title = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
attr = json.loads((REPO / "data/clips/attribution.json").read_text())[clip]
program = json.loads((pkg / "program.json").read_text())
n_static, n_obj = len(program.get("static", [])), len(program.get("objects", []))
moving = [o["id"] for o in program.get("objects", []) if (o.get("motion") or {}).get("type", "static") != "static"]
dur = program["meta"].get("duration", 0)

(pkg / "README.md").write_text(f"""# {title}

这是 GameWorldModel 从一段 {dur:.0f} 秒的视频自动生成的一个小游戏。整个过程没有人工建模，视频进去，场景程序和可玩的游戏出来。项目主页 https://github.com/jwj1342/GameWorldModel

## 先看视频

side-by-side.mp4 是最直观的一个。左边是原视频，右边是生成出来的游戏用同一条相机轨迹渲染的画面，可以直接对着看恢复得准不准。

source.mp4 是喂进去的原视频。overlay.mp4 把感知结果画回到视频上，能看到每个物体被框成什么样、判成什么运动。gameplay.mp4 是自动试玩录下来的，一个蓝色胶囊从出生点走到绿色的目标框。

## 自己玩

需要装了 Python 的机器，然后

    python3 serve.py

浏览器会打开 http://localhost:8000/ 。WASD 或方向键移动，空格跳，R 在回放和游玩之间切换。左上角显示收集了多少个物件、死了几次、有没有到终点。

不想用 Python 的话，在 game 目录里起任何静态服务器都行，比如 npx serve 或者 php -S localhost:8000。不能直接双击 index.html，浏览器不让本地文件加载 ES 模块。

## 里面有什么

program.json 是这个游戏的场景程序，一份 JSON，里面是地面和静态结构、每个物体的几何和位置、每个物体的运动方式，以及玩法绑定的出生点和目标点。改里面的数字再刷新页面，游戏就变了。这份程序是模型看着视频关键帧和感知结果写出来的，不是人写的。

这个场景有 {n_static} 个静态结构和 {n_obj} 个物体{"，其中 " + "、".join(moving) + " 会动" if moving else ""}。

report.md 是这次运行的完整报告，包括感知阶段认出了什么、模型分几步写了什么、渲染比对的分数、自动试玩的结果。

game 目录是可以独立运行的游戏，里面的 kernel 是固定的运行时代码，vendor 是 three.js 和 Rapier 物理引擎。

## 出处和许可

{attr["note"]}

原视频 {attr["title"]}，作者 {attr["author"]}，许可 {attr["license"]}{"（" + attr["license_url"] + "）" if attr["license_url"] else ""}{"，来自 " + attr["source_url"] if attr["source_url"] else ""}。

代码部分是 GameWorldModel 项目的一部分。three.js 和 Rapier 的许可见 game/vendor 下各自的文件。
""")

(pkg / "ATTRIBUTION.md").write_text(f"""# 出处与许可

## 视频

标题 {attr["title"]}
作者 {attr["author"]}
许可 {attr["license"]}{" " + attr["license_url"] if attr["license_url"] else ""}
来源 {attr["source_url"] or "本项目自行渲染"}

{attr["note"]}

本包里 source.mp4 是原视频的截取转码版本，overlay.mp4 和 side-by-side.mp4 含有原视频画面。replay.mp4、gameplay.mp4、program.json 和 game 目录是程序生成的内容，不含原视频像素。

## 代码与依赖

game/kernel 和 program.json 属于 GameWorldModel 项目 https://github.com/jwj1342/GameWorldModel

game/vendor/three 是 three.js r186，MIT 许可。
game/vendor/rapier 是 Rapier 3D 0.20，Apache-2.0 许可。
""")
print(f"  docs written for {clip}")
