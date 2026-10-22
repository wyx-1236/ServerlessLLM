# ServerlessLLM Baseline 仓库建立操作指南

本文是[完整升级方案](engine-upgrade.md)的工程建立步骤，不是已执行记录。只建立用户的
SLLM Fork 和独立 checkout，不修改 CrossPool 的 remote 或既有研究目录。
下列命令分步骤执行、逐项检查退出状态，不能将整页作为无检查的批处理脚本。

## 1. 选择账号和确认来源

官方来源是 [ServerlessLLM/ServerlessLLM](https://github.com/ServerlessLLM/ServerlessLLM)。
推荐 Fork 到自己的 GitHub 账号，名称保留 `ServerlessLLM`；已有合适 Fork 则复用。
组织 Fork 需要该组织的创建权限。执行前确定 owner、仓库名称和本地目标目录。
这些是执行参数，不是需要重新讨论的引擎设计决策。

使用已有登录会话或 GitHub CLI 的认证状态核验账号；没有登录时完成 GitHub 正常
登录流程，不把 token 贴进聊天或命令参数。先确认是否已有同上游 Fork，避免重复。

```powershell
git --version
gh --version
gh auth status
```

## 2. 建立或复用 Fork

网页方式：打开官方仓库 → Fork → 选择 Owner → 名称 `ServerlessLLM` →
Create fork。默认只复制默认分支即可；固定历史提交在后续 fetch 后核验。
不要点击新建空仓库后手工上传源码，否则会丢失 Fork 关系和提交祖先。

也可以在已核验的个人账号下使用 CLI，禁止在 CrossPool 目录让工具自动改 remote：

```powershell
gh repo fork ServerlessLLM/ServerlessLLM --clone=false
```

显式给定仓库参数时不传 `--remote=false`，本机 GitHub CLI 会拒绝该组合；
`--clone=false` 已避免自动 clone。组织场景增加 `--org` 并传入选定的真实组织名。
若仓库已存在，先读取其
metadata，确认 `isFork=true`、parent 是官方仓库、有当前账号写权限，再继续。
同名且 parent 不符时不覆盖、不删库，重新选择名称或正确 Fork。

Fork 会继承官方源码。后续发布的改动和文件按仓库可见性对外可见；这里只纳入
可发布的源码、配置和轻量证据摘要，保留许可证，不上传机器凭据或权重。

## 3. 独立 clone 与远程设置

本机建议目标目录为 `<workspace>\ServerlessLLM-baseline`，
与 CrossPool 和历史准备目录并列。它是拟建立的目录，不代表已经存在或已 clone。
若目标已存在，先只读检查其 Git 根目录、HEAD、remote 和未提交文件；只有确认是
同一任务的适合 checkout 才复用，不删除、不 reset、不自动 stash。

以下输入真实 Fork URL 和绝对本地路径；clone 目标应尚不存在：

```powershell
$sllmForkUrl = Read-Host '已核验的 Fork HTTPS clone URL'
$sllmCheckout = Read-Host '新 checkout 的绝对路径'
git clone $sllmForkUrl $sllmCheckout
Set-Location -LiteralPath $sllmCheckout
git remote add upstream https://github.com/ServerlessLLM/ServerlessLLM.git
git remote -v
git status --short
git fetch upstream --tags
```

`origin` 必须指向用户 Fork，`upstream` 指向官方仓库。已有 remote 时先核对，
不盲目重复 add 或改写。在源准备目录中的旧 `.git` 不作为本次新建仓库的替代品。

## 4. 核验参照提交并建立分支

普通完整 clone/fetch 后先核验提交存在；若历史不可用，再从官方完整历史或已核验
bundle 获取，不把当前 latest 代替旧参照。执行新建操作前确认同名分支/tag 是否已
存在；存在时比较指向，符合则复用，不符合则报告冲突，不覆盖引用。

```powershell
$sllmBaseCommit = '9f50241baa5386e06a9321c51f19a9ef5f964c2b'
git cat-file -t $sllmBaseCommit
git show --no-patch --format=fuller $sllmBaseCommit
git branch --list 'codex/baseline-vllm' 'codex/vllm-upgrade'
git tag --list 'sllm-baseline-upstream-9f50241'
```

首次建立、且同名引用均不存在时：

```powershell
git switch -c codex/baseline-vllm $sllmBaseCommit
git tag -a sllm-baseline-upstream-9f50241 $sllmBaseCommit -m 'Unmodified SLLM upstream reference'
git switch -c codex/vllm-upgrade
git rev-parse HEAD
git log -1 --oneline
git status --short
```

此时两个新分支都从同一原版提交起步；candidate02 和研究改动尚未导入。
确认 Git 作者身份已有正确配置，不擅自更改用户全局身份。

## 5. 首次发布引用与 PR 方向

当正式实施范围已包含发布至该 Fork，核验 origin 和引用指向后发布：

```powershell
git push origin codex/baseline-vllm
git push origin refs/tags/sllm-baseline-upstream-9f50241
git push -u origin codex/vllm-upgrade
```

执行授权只包含本地准备时，停在本地引用，不自行发布。不用 `--force` 或批量
`--tags` 推送其他历史 tag。分支/tag 发布后读取远程引用，确认与本地 SHA 一致。
PR 有实际改动后再创建，不能在两个完全相同的分支之间先声称 PR 已建立。

| PR 字段 | 明确选择 |
| --- | --- |
| base repository | 用户自己的 Fork |
| base branch | `codex/baseline-vllm` |
| head repository | 同一 Fork |
| compare branch | `codex/vllm-upgrade` |

GitHub 可能默认将 Fork PR 指向官方仓库，因此必须逐项检查。候选阶段用 draft，
写明实际完成范围和缺失验收；通过后再审查与合并。无需先向官方提交升级方案。

## 6. 导入设计与最小改动

读取新工程自己的说明和约束，迁入主方案要求的 `docs/baseline/` 设计、provenance
与证据索引，再分层引入研究中可复用的代码。不要整体复制准备目录覆盖 checkout，
也不要把 Transformers candidate-stable 当作旧 vLLM 实测基线。

每项迁入记录源文件/补丁摘要、用途、采用或排除原因及对应测试。先有依赖锁和
无 GPU 契约测试，再申请 Store/服务 GPU 实验。源码提交、hooks、GPU审批及释放
遵循主方案与实际工程约定；这里的仓库建立不涉及 GPU 使用。

## 7. 工程建立核对表

- 实际 Fork URL 可访问，parent 是官方仓库，当前账号可写。
- 独立 checkout 根目录与 origin/upstream 正确，既有研究目录原样保留。
- 原版 tag 与集成分支初始 SHA 均为固定参照提交；升级分支来源可追踪。
- 设计、provenance、证据索引与后续测试/部署入口的目标位置明确。
- 首次发布的引用与本地一致；尚未发布的引用如实标明，不能伪报远程完成。
- 有实际改动后 PR 指向自己的集成分支，不指向官方或 Fork 最新默认分支。
- Linux 实验部署按确定 commit/tag；权重、依赖缓存和结果另存，未启动任何 GPU 作业。

核对后提交阶段 0 报告，列出实际 Fork、本地路径、远程引用/commit、验证结果、
未完成项和下一阶段范围。等待用户明确审查通过后，才进入环境冻结和代码适配；
不能因为仓库建立成功就自动继续。报告及审查规则以主方案的“阶段间的具体
交付”为准。

## 官方操作参考

- [GitHub Fork 建立与 clone](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo)。
- [origin/upstream 配置](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/configuring-a-remote-repository-for-a-fork)。
- [GitHub CLI `gh repo fork`](https://cli.github.com/manual/gh_repo_fork)。
