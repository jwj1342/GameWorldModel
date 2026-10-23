# 记录第二段视频重建效果

## 本次提交

关联 #3、#32，记录第二段桌面视频的缓存派生重建结果，补充图片、视频和实验说明。本 PR 仅提交效果材料及文档，通用修复源码另行整理；不关闭 Issue，也不替代 #80 的第一段自拍视频记录。

## 重建效果

下图从左到右依次为原视频、修复前、修复后，展示前、中、后三个时刻。

![原视频、修复前与修复后三时刻对照](https://github.com/jwj1342/GameWorldModel/blob/codex/video2-progress-9404/docs/results/assets/video2-9404-comparison.jpg?raw=true)

[查看8秒原视频／重建对照视频](https://github.com/jwj1342/GameWorldModel/blob/codex/video2-progress-9404/docs/results/assets/video2-9404-comparison.mp4) · [查看重建视频](https://github.com/jwj1342/GameWorldModel/blob/codex/video2-progress-9404/docs/results/assets/video2-9404-reconstruction.mp4)

视频为192帧、24FPS，由实际浏览器渲染帧编码。对照视频左侧为输入，右侧为重建；正文直接展示对照图，不依赖管理员访问本地或SOAR。

## 修复与实际变化

- 缺少可靠地面候选时保留 ground=unknown，不再把正对相机的背景大面当成地面并旋转整个世界。
- 不自动补绿色地面或接地关系；杯子、花瓶和蜡烛的整体朝向得到改善。
- 保留已有竖屏尺寸和完整相机内参传递修复，复用5个资产引用；本轮不重新感知或检索资产，也未手动移动物体。
- 未知地面进入纯场景预览，禁止自动推断玩家、目标及物理玩法；Writer与接触Repair入口增加对应保护。

## 验证

- CPU作业9404：Evidence校验0错误、Quality Gate为warn；编译0错误/0警告，浏览器error/pageerror/requestfailed为0。
- 最终CPU回归9405：Python 953 passed、5 skipped，Node 69 passed。跳过项涉及当前环境缺失的真实资产条件，不计为通过。
- 保留10条对象轨迹，源缓存哈希前后相同；视频文件下载后SHA256一致。最终回归与视频使用的关键视觉生产源码逐字节一致。
- 本轮模型/API/GPU调用均为0。源感知来自9373；本轮是该有损缓存的派生诊断，不是新的端到端重建实验。

## 尚未解决

书本仍有直立、堆叠错误，杯垫偏厚，材质和细节较粗糙。真实地面、重力和米制尺度未确认；当前无物理交互，不能称为高保真重建或可玩性成功。

本结果不计入 Issue #3 正式对照，不改写历史失败结论。请优先审核地面不确定性处理及下游边界是否合理，后续再处理资产朝向和尺寸问题。

## 实验身份

源码快照基于 a670c182c5c3bb7d94bea8315a3ce9782f1c2e24，包含独立本地修复；该快照不随本PR提交。感知缓存来自9373，重建与渲染为9404，最终CPU回归为9405，不能当作本PR文档分支或main的测试结果。

对照视频SHA256：b2c53c7a1933b15a4583dcdb6e3239c325a514309bf9cd17a21a97cf3190f9f6。

重建视频SHA256：039c16bf38d99f3b99210b7807b5be76f4d199df47c9216cc815a0e5c9ce3842。

