# 原版功能与规范核查：迁移续写及升级兼容边界

核查日期：2026-10-23。固定上游基点为
`9f50241baa5386e06a9321c51f19a9ef5f964c2b`，被核查升级候选为
`722160025ab0131bd865502cb9cfd5e6b9959430`。本文区分项目承诺、源码实际行为、
CPU 受控观察与已归档 GPU 结果；不认领全项目、旧引擎或所有后端已完成实机回归。

本次仅核查和修订文档。未修改生产代码、依赖或模型产物，没有调用 GPU。
[修订兼容方案](migration-continuation-proposal.md)仍待用户审查。

## 核查结论

1. **迁移中继续完成用户请求是原项目的功能意图，不是可随升级删除的选项。**
   README 声明零停机迁移；官方迁移文档明确展示抢占后在目标实例恢复请求。
   官方示例配置使用 Transformers，不能将其直接当作旧 vLLM 已具备相同能力的证据。
2. **固定版本的两个后端实现不同。** Transformers 有抢占信封和
   `resume_generate`；旧 vLLM 仅在 `shutdown` 中 abort，既没有该方法，也不返回
   Router 所需的抢占信封。抽象基类甚至未要求此方法，说明存在项目意图与后端实现
   不一致的继承缺口。不能据此降低升级后的项目级功能要求。
3. **新版 D1 修复了可运行性和生命周期，但没有补齐请求续写。** 真实 token
   重放、缓存、路由切换及两类回滚已验证；3 个源活动请求仍以 abort 结束。
   这既不是透明续写，也不能证明与旧引擎的 GPU 终态等价。
4. **无回归结论目前还受其他兼容缺口阻断。** 旧 Store CLI 导出路径尚未接入
   新准备流水线；下载参数接受行为、调试快照行为有可证明差异；共享路径与新版依赖
   下 Transformers/LoRA/微调/embedding 的完整回归证据不足。

## 证据方法与限制

从本地不可变 Git 对象读取 31 个文档、示例、源码、测试和依赖文件，逐文件保存
SHA256；其中 21 个文件逐字节一致。字节一致只能证明该文件未改，不能证明其调用方、
共享组件或新版依赖无影响。未切换工作区、未应用历史补丁。

CPU 对照使用 AST 提取的真实方法体与最小后端/Ray 替身，26 项通过，2.41 秒。
为到达原 Router 的深层协议，测试显式提供 `ready_instances` 别名，绕过已确认的
原实例池名称缺陷；此处理只用于分层核查，不证明原样完整 Router 能启动。
升级候选既有迁移、生命周期、旧行为及 CLI focused 测试另有 69 项通过，20.23 秒。
两次审计替身缺少字段/标准库变量的初始失败已保存，修复的是审计替身，不是生产代码。

这些测试没有执行真实引擎、真实 Store、原版完整服务或 GPU。现有
[D1 归档](evidence-index.json#stage_d1_runtime_qualification)是新版限定运行证据。
本轮服务器只读依赖源码核对因 SSH 超时失败，不能据缓存推断实际依赖兼容性。

## 原版迁移行为：逐环节对照

| 环节 | 固定上游的实际实现 | 升级候选及证据 | 兼容结论 |
| --- | --- | --- | --- |
| 迁移触发 | StorageAwareScheduler 按存储/迁移延迟规划，Router 创建目标 | 保留 planner；D1 用显式零权重 fixture 触发真实队列和迁移 | fixture 不是原示例的真实第二模型，后者仍需回归 |
| 源 token 采集 | `list[list[int]]`；vLLM 每请求仅拼接 prompt 与第一个输出候选；无请求 ID 对应关系 | 仍取首候选；活动注册提前，历史与活动分离 | 不能支撑多候选通用续写，不能用列表位置代替请求身份 |
| 重放迭代 | 只按第一序列长度计算 delta；达到阈值时先退出，不重放该轮 | D1 保留该算法 | 这是预热过程，不能充当最终一致快照 |
| Transformers 预热 | 单 token generate，保存 `past_key_values` 和含预热输出的 `current_tokens` | 文件逐字节未改；新依赖实机兼容未验证 | 必须保留该后端路径，不能要求它改用新 vLLM 信封 |
| Transformers 抢占 | streamer 在 DELETING 时抛异常，generate 返回 `preempted/current_output/completed_tokens` | Router 仍接收这些字段，复制预算字典后调用原方法；CPU 对照一致 | 项目存在明确的请求接续协议意图 |
| Router 接续 | 从 `migration_record` 找目标；`max_tokens` 减已完成数，再调用 `resume_generate` | 字段/调用形状保持；输入字典不再原地修改 | 保持用户预算语义，字典副作用变化应显式记录 |
| 旧 vLLM 抢占 | shutdown abort 跟踪 ID，generate 将引擎输出格式化；没有抢占信封或 resume_generate | 普通 shutdown 仍是强制关闭；D1 观察到 3 条 abort | 继承缺口未补齐，不能作为升级验收终点 |
| 旧 vLLM 重放 | 对每序列 `max_tokens=1`；返回 None；丢弃 generate 的错误体 | 成功仍返回 None，错误体改为抛异常以触发回滚 | 错误传播是显式纠错，不能写成原版已有的保障 |
| 路由/回收 | 发布目标、记录映射、shutdown/kill 源并归还预留；多处失败缺少兜底 | 请求/actor/预留所有权、取消和失败回收已修复 | 原构造参数/实例池错误是继承问题；C 的 owner 登记遗漏是升级回归，两者分列 |
| 用户响应 | 非流式请求最终返回响应；vLLM 多 choices 和 usage 聚合 | CPU 对照 response ID、choices、usage 一致，D1 新 HTTP 一致 | 新 HTTP 成功不能代替原活动请求成功；不新增外部重发要求 |

### 原版续写并不自动等于严格数值等价

原 Transformers 的目标续写在预热序列更长时，会使用目标 `current_tokens`
替换源最终输出，再按 Router 扣减后的预算生成。CPU 方法对照构造“原预算 4、
源已提交 2、预热多 1、目标再生成 2”的情况，真实方法返回 5 个 completion token。
这是受控方法级反例，不是完整模型实测；它证明不能机械照搬预热输出拼接，或声称
旧方法已保证预算严格正确。原 finish_reason 也用总 completion 与剩余预算比较，
存在边界风险。原 `test_resume_generate` 只断言结果存在且无 error，不检验逐 token
等价、预算、停止条件或完整 usage。

修复这些问题应保持“原请求能继续完成且遵守用户参数”的功能契约；不能将复现缺陷
作为正确兼容的定义，也不能隐瞒这种修复与旧实现的差异。

## 全项目功能和规范影响面

| 功能/规范 | 原版依据与当前状态 | 修订后的保留及验证要求 |
| --- | --- | --- |
| HTTP/CLI | 原 app_lib、CLI 文件未改；注册/更新/删除、生成、embedding、微调任务和模型列表仍存在 | 保持 endpoint、有效配置、响应字段和已有错误协议；覆盖更新/LoRA/微调路径，不能只测生成 |
| Transformers 迁移 | 官方示例及后端具备接续意图；共享 Router 已变，后端文件未变 | 在兼容依赖下回归抢占、原信封、目标续写、资源计数和官方双模型场景 |
| vLLM 输入/采样 | 原 `input_tokens > prompt > messages`，messages 按 role/content 拼接；SamplingParams 转发请求参数；支持多 choices 响应 | 保持原 vLLM 提示构造，不改成 Transformers chat template；覆盖原有效采样参数与多候选，不以永久 n=1/贪心限制交付 |
| stop/shutdown | stop 排空；shutdown 强制 abort；管理删除/取消与迁移不是同一件事 | 新增仅用于迁移的内部交接流程，不将普通 shutdown 改成自动续写或将用户取消恢复 |
| 调试与活动快照 | 原 trace_debug 保留完成记录，原测试会在完成后读取这些 token；新版只返回活动记录，另存 request_history | 存在实际差异。活动迁移不能混入历史；兼容诊断视图、测试变更与调用方迁移须显式设计，不可声称原测试原样通过 |
| Store 导出入口 | 主包 downloader 已走新准备链；原 `sllm-store save --backend vllm` 和导出 example 仍依赖旧 patch 导出方法 | 所有对外导出入口必须适配并保持参数/路径优先级；主包导出通过不等于 CLI 已通过 |
| pattern/max_size | 原接口接受并层层转发，但固定 patch 的 save_model 未使用它们；新版主包直接 ValueError | 确认参数接受行为退化，不能声称旧版真实实现了过滤/分片。保留旧可调用性及无实际效果的事实；新增语义另审，不静默拒绝 |
| 旧 Store 产物/新格式 | 新增身份、张量/alias/buffer 与完整性验证；仅新版冻结产物已实测 | 提供旧产物辨识与明确转换/兼容路径，不覆盖旧数据；不能要求用户手动绕过 Store 或直接 HF 加载替代 |
| LoRA/微调/embedding | README、功能文档和 API 有相应能力；依赖升级可能影响原后端 | 单列受影响后端/配置回归和数据格式；narrow vLLM 运行结果不代表它们已经通过 |
| 自动伸缩/调度/回收 | 轮询、storage-aware、placement、并发目标、scale-to-zero；共享控制面有修改 | 验证 min/max/target、队列、取消、部分迁移和资源 owner；不改变原调度目标规避续写 |
| 环境/部署/硬件 | 原项目有 NVIDIA、实验性 ROCm 与 Docker/多节点路径；新版限定 Linux/CPython/CUDA 组合已冻结 | 内部版本可按已批准方案适配，但旧有部署入口不能被悄然破坏；镜像/其他平台仍须其适用范围证据，不能泛称全项目已支持 |

原文档还明确写过 storage-aware 只支持 Transformers，而代码存在 vLLM 分支；
原迁移示例也与固定版本的必需构造参数存在不一致。文档、示例、实现有历史漂移，
需逐项标注，不能将文档示意日志当作当前固定版本完整实测证据。

## 修订后的决策与交付关卡

采用“既有项目契约不变、vLLM 后端内部适配、最小共享改动”的方向。保留
Transformers 的现有抢占信封及接续调用；为 vLLM 增加内部最终快照/交接能力，
而不是向所有后端增加一个新的必选抽象方法。详见[修订方案](migration-continuation-proposal.md)。

先闭合入口/参数/调试语义和共享依赖的兼容证据，再审查代码实施计划。随后以 CPU
验证建立请求及资源所有权证明，另行申请精确双卡完成真正的“同一 HTTP 请求”迁移
输出对照，并回归原 Transformers 双模型场景。贪心单候选可以是第一批实验输入，
不能成为最终功能范围的删减。未证明的项目功能继续阻断完整兼容声明和正式发布。

## 固定来源

- [README 迁移承诺](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/README.md#L198)
- [官方迁移示例与恢复日志](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/docs/features/live_migration.md#L118)
- [示例的 Transformers 配置](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/examples/live_migration/config-qwen-3b.json#L3)
- [原 Router 接续分支](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/routers/migration_router.py#L49)
- [原 Transformers 抢占](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/transformers_backend.py#L278)及[目标接续](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/transformers_backend.py#L422)
- [原 vLLM abort、快照及重放](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm/backends/vllm_backend.py#L265)
- [原续写测试的断言范围](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/tests/backend_test/transformers_backend_test.py#L254)
- [原 Store CLI 导出路径](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm_store/sllm_store/cli.py#L62)及[原 patch 参数实现](https://github.com/ServerlessLLM/ServerlessLLM/blob/9f50241baa5386e06a9321c51f19a9ef5f964c2b/sllm_store/vllm_patch/sllm_load.patch#L160)

可移植证据摘要见[evidence index](evidence-index.json#original_function_audit_20261023)。
外部核查源快照、SHA256、方法级测试、JUnit 和失败回执保存在
`engine-upgrade-20261022/stageD1/original-contract-audit-20261023`，由个人报告链接。
