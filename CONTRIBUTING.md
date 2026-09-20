# 怎么参与

main 分支开了保护，不能直接往上推，改动都通过 Pull Request 进来，需要一个人批准之后才能合并。

## 流程

```bash
git checkout -b 你的分支名        # 起个能看出在做什么的名字，比如 fix-motion-classification
# 改代码
python -m pytest -q tests        # 开 PR 之前先跑一遍，11 个测试应该全过
git commit -am "一句话说清楚改了什么"
git push -u origin 你的分支名
gh pr create                     # 或者去网页上开
```

PR 里写清楚改了什么、为什么改。如果动了感知或者生成那部分，最好附上改动前后跑同一段视频的对比，`docs/status.md` 里有当前的分数可以对照。

评论都解决掉、有一个批准之后就能合并。

## 提交之前注意

不要提交 API key。`.secrets/` 已经在 `.gitignore` 里了，但还是自己确认一下。

不要提交视频和运行产物。`data/clips/` 和 `out/` 也都忽略了。想分享结果的话用 `scripts/collect_results.sh` 把该看的挑到 `docs/results/`，那个目录是进 git 的。

依赖有变动的话，`requirements.txt` 是给所有人用的通用列表，`requirements-vulcan.lock.txt` 是集群上的精确版本，两个都要更新。

## 环境怎么搭

看 [RUNNING.md](RUNNING.md)，里面分了在自己电脑上跑和在 HPC 上跑两种情况。

只改场景程序、编译器或者 three.js 内核的话，不用下那 9.6 GB 的感知权重，装好 Python 依赖和 `npm install` 就能用 `examples/handwritten/program.json` 调试。
