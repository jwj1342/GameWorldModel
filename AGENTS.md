# 给编码智能体的说明

这是一个研究项目：让编码智能体把真实场景写成能运行、能交互的代码。项目现在在做什么、哪些部分已经冻结，看 `docs/status.md`。

动手之前先读 `CONTRIBUTING.md` 里的三节：「工程上守的几条」「新东西放哪」「模块之间的边界」。那里的规则对智能体同样适用，这里不重复，只列最容易踩的几条：

- 改完跑测试：`python -m pytest -q tests`，Node 部分 `node --test tests/*.mjs`。
- 新功能和新算法进 `gwm/<包>/` 并配测试；一次性的探针进 `scripts/probes/`；实验结论登记到 `experiments/index.csv`；相关工作登记到 `related-work/index.csv`。
- 运行产物写进 `work/`，不要在仓库外另开目录，也不要提交它们。API key、视频、PDF、会议转写都不进 git。
- 先写最直接的版本：不为没出现的需求加参数、抽象和兜底，不写吞掉错误的 try/except。
- 仓库是公开的：issue、PR、文档里只写思路和数字，不写人名。
- 命名按内容起，不用 V0、M1、v2 这类版本号。
- 在 Vulcan 集群上：登录节点不跑 Python、不编译，计算都进 Slurm 作业；计算节点的代理不放行 npm 和 arXiv，这类下载在登录节点做。
