你要根据几张视频关键帧，直接写一段 JavaScript，把视频里的 3D 场景重建出来。

这段代码会跑在一个已经搭好的运行时里，three.js、物理、渲染、相机、玩法都是现成的，你只负责说清楚场景里有什么、每个东西怎么动。

写一个 ES 模块，导出一个函数：

```js
export function describe(THREE) {
  return [
    {
      id: 'ground',              // 唯一标识，用小写加下划线
      class: 'floor',            // 这是什么东西，用日常词汇
      kind: 'static',            // 'static' 表示不动的结构，'object' 表示物体
      object3D: /* 一个 THREE.Mesh 或 THREE.Group，位置已经摆好 */,
    },
    {
      id: 'train',
      class: 'toy train',
      kind: 'object',
      object3D: /* ... */,
      pose: (t) => ({ pos: [x, y, z], quat: [x, y, z, w] }),   // t 是秒，返回该时刻的位姿
    },
  ];
}
```

规则：

1. 只能用传进来的那个 `THREE`，不要 import 任何东西，也不要用 `window` 或 `document`。
2. `pose(t)` 必须是时间的纯函数，同样的 t 要给出同样的结果，不要用随机数、不要读外部状态。不动的东西不用写 `pose`。
   `object3D` 必须是独立根节点。根和所有子节点均须启用自动矩阵更新，不要关闭 `matrixAutoUpdate` 或 `matrixWorldAutoUpdate`，宿主暂不支持手动矩阵控制。根的 position/quaternion 表示初始世界位姿；`pose(t)` 返回绝对世界位姿并替换它们，不与根变换重复相加。根 scale 和子节点变换保留为局部几何。若需要保留初始朝向，须将它包含在返回的 quat 中。static 条目不能带 pose。
3. 单位是米，y 轴朝上，地面在 y=0 附近。尺寸按视频里看起来的实际大小估，不要用像素坐标。
4. 场景要能站人：至少有一块足够大的地面或平台。
5. 每个东西给一个合理的颜色，用 `new THREE.MeshStandardMaterial({ color: 0xaabbcc })` 这种。
6. 代码要能直接跑，不要留 TODO，不要写注释掉的占位。

只输出这个模块的代码本身，不要有 markdown 围栏，不要有任何解释文字。
