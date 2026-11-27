# 任务：把一个真实房间写成一份可执行的 three.js 场景

你拿到的是同一个真实房间的 {{VIEW_COUNT}} 张照片，以及每张照片的相机位姿。请把这个房间写成代码。

写出来的场景会从两个方面被评测，两个分数分开报，不合成一个：

1. **像不像。** 用你**没见过的**另一批相机位姿渲染你的场景，和那些位置拍的真实照片比。所以要对的是三维结构本身，只在某几张照片的视角下看着对是没用的。
2. **能不能动。** 评测方有一份对着真实房间写的交互清单，你看不到。它会在物理引擎里真的去执行：推得动的东西是不是推得动，推不动的是不是推不动，放在桌上的东西松开之后是不是还在原地，门是不是绕着轴转。光把物体声明成能动不算数，要执行出来成立才算。

## 你手上有什么

```
inputs/views/*.jpg      {{VIEW_COUNT}} 张训练视角的照片
inputs/cameras.json     每张照片的相机：pos、quat [x,y,z,w]、fov_deg（竖直方向）、aspect
{{POINTS_INPUT}}inputs/frame.json       坐标系说明和房间大致范围
{{MANIFEST_INPUT}}tools/preview.sh        渲染你的场景并和照片并排，用来自己检查像不像
{{SCORE_TOOL}}tools/physics.sh        在物理引擎里把你的场景执行一遍，用来自己检查能不能动
```

坐标系：**y 轴朝上，地面在 y = 0，单位大致是米**，墙大致沿 x 轴和 z 轴。相机是 three.js 的约定（x 右、y 上、看向 -z）。尺度是按拍摄者手持相机的高度估的，不是量出来的，所以{{SCALE_SOURCE}}，不要套用你印象里家具的标准尺寸。

{{POINTS_HINT}}

## 你要交什么

在当前目录下写两个文件。

### `scene.js`

```js
export function describe(THREE, ctx) {
  // 返回一个数组，每个物体一项
  return [
    { id: 'floor', kind: 'static', object3D: ... },
    { id: 'coffee_table', kind: 'movable', mass_kg: 8, object3D: ... },
  ];
}

// 可选：背景色和灯光。不写就用默认的一盏天光加一盏平行光
export function environment(THREE) {
  return { background: '#d8d4cc', lights: [ /* THREE 的灯 */ ] };
}
```

每一项的字段：

{{ID_RULE}}
- `kind`：只有两种。`'static'` 是这个房间里搬不动的东西；`'movable'` 是人能徒手挪动的东西。这是你要对着照片做的判断。
- `object3D`：一个 `THREE.Object3D`，通常是一个 `Group`，里面放若干个基本体网格（盒子、圆柱、球、平面等）。它自己的 `position` 和 `quaternion` 就是这个物体在世界里的位姿。物体由哪些部件组成，就拆成哪些网格：桌子是一块面板加几条腿，不是一个实心方块。
- `mass_kg`（可选）：能动的物体的大致质量。
- `interaction`（可选，只给能动的物体）：不写就是可以自由拿起来拖动。门、抽屉这类有约束的东西要写：
  - 绕轴转：`{ grab: 'hinge', axis: [0, 1, 0], pivot: [x, y, z], limit: [0, 1.57] }`，`pivot` 是转轴经过的一个世界坐标点，`limit` 是弧度。
  - 沿轴滑：`{ grab: 'slide', axis: [1, 0, 0], limit: [0, 0.4] }`，`limit` 是米。

规则：

- `describe` 每次被调用都要**新建**所有物体，不要把物体缓存在模块顶层。
- 文件里不能 `import` 任何东西，要用的都从参数 `THREE` 和 `ctx` 拿。
- 碰撞体是按每个网格的包围盒自动算的，你不用写。所以几何摆在哪里，物理就在哪里：物体悬空会掉下去，和别的东西插在一起会被弹开。把东西放在支撑面的正上方，留一两毫米的缝。
- **不许把照片或者照片里的像素当贴图用。** 所有材质都必须通过 `ctx.material(名字)` 拿，不要自己 `new` 材质，也不要加载任何图片。

### `materials.json`

场景里每一种表面写一条。外观在这里是一段文字加一个随机种子，评测时由一个文生图模型按这段文字生成可平铺的贴图；另外给一个底色，用于不贴图时的纯色渲染。

```json
{
  "oak_floor": {
    "prompt": "light honey-coloured pine floorboards, long narrow planks, visible knots",
    "seed": 1,
    "base_color": "#c8964f",
    "roughness": 0.6
  }
}
```

- 名字只能用小写字母、数字和下划线。
- `prompt` 用英文，描述这块**材质本身**长什么样（材料、颜色、纹理的粗细和方向），不要描述物体或者场景。
- `base_color` 尽量贴近照片里这块表面的平均颜色。纯色渲染时它就是全部外观，贴图渲染时也要靠它判断贴图色调对不对。
- 在 `scene.js` 里用 `ctx.material('oak_floor', { repeat: [4, 6] })` 取材质。`repeat` 是贴图在这个网格的 UV 上平铺几次，按表面的实际大小估：一张贴图大约对应现实里一米见方。

## 怎么检查自己的工作

```bash
./tools/preview.sh                      # 默认挑几张训练视角，输出 preview/*.jpg（左边照片，右边你的渲染）
./tools/preview.sh DSCF4668,DSCF4775    # 指定视角
./tools/physics.sh                      # 输出每个能动的物体抓不抓得起来、放开之后稳不稳
```

用 Read 工具打开 `preview/*.jpg` 看。重点看：家具的位置和大小有没有对上照片，地面和墙的交线对不对，有没有漏掉的大件。`tools/physics.sh` 报出不稳或者拖不动的物体时，多半是它悬空了、陷进了支撑面，或者和邻居插在一起。

建议的顺序：先搭地面、墙和大件家具，预览对齐之后再加小物体，最后调颜色、材质和灯光，全程隔一阵就跑一次物理自检。{{POINTS_SCRIPT_HINT}}

## 什么时候算完成

{{MANIFEST_DONE}}
- `./tools/preview.sh` 能跑通，画面里主要家具的位置和大小与照片大致吻合。
- `./tools/physics.sh` 里你声明能动的物体都通过。
- 最后用几句话说明：哪些物体你判成了能动、哪些判成了不能动、还有哪些地方你知道没做好。

{{MANIFEST_SECTION}}
