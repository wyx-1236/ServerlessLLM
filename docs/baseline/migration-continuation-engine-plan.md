# vLLM 请求状态恢复与迁移交接增补

## Goal 与审查边界

补齐 [迁移续写方案](migration-continuation-proposal.md) 尚未落实的引擎恢复边界，
使同一外部请求在迁移后继续完成。保持原调度算法、Store 加载与迭代预热、请求
字段及多候选/采样/停止语义。此增补提出跨 frontend、EngineCore 和 Worker 的
固定版本恢复适配，落实原方案的后端内部恢复边界；不授予 GPU 使用权限，也不将
较窄的贪心单候选用例定义为最终支持范围。改变引擎、runner 或调度算法仍须审查。

## Baseline 与已排除的捷径

固定 vLLM 0.31.0 的 `EngineCoreRequest` 不传已生成历史；`Request` 构造从空
`output_token_ids` 开始。V1 `NewRequestData` 不含输出历史，GPUModelRunner
创建 `CachedRequestState` 时也从空输出开始。仅修改 `scheduler_cls.add_request`
不能使 Worker、采样器和 frontend 获得一致的恢复状态。

`resumable`/streaming input 将前次输出并入 prompt 并重置生成预算，不是保留
prompt/output 分界的恢复接口。V2 `trace_replay` 会覆盖预算、忽略 EOS、清除
stop 且限制 n=1，也不能替代正常请求续写。不能把这两条路线包装成完整兼容。

“把源输出接到目标 prompt 后重新 generate”改变 presence/frequency penalties、
min_tokens、stop 字符串跨边界匹配及部分 logprobs/usage 语义，不能作为最终实现。

## 推荐变更

在固定 vLLM 0.31.0 上提供显式的请求状态恢复入口，同时恢复三层状态。
采用 V2 model runner 的 partial-output prefill：保留原 prompt_len，将完整
prompt/output 历史交给既有 Worker 初始化；其采样器按两段恢复 penalty 与
min_tokens 状态。V1 engine 不等于 V1 model runner，不能混用术语。固定版本
Scheduler 子类仅在请求准入时附加历史，保留原同步/异步 Scheduler 的调度方法。
frontend 通过 OwnedEngine 所有的实例适配，在创建状态与发送 Core 之间恢复
detokenizer/logprobs；Core 提交确认用于避开迟到注册竞态，不安装全局 monkeypatch，
也不修改上游 wheel。

对实际不使用 V2 的实例，冻结前明确拒绝迁移，源继续服务；不得偷偷改变 runner
或丢弃请求。这样的拒绝是开发保护，不能用于声明所有配置迁移已完成。原 Store
Worker extension 保持装载/导出职责，Store 产物不覆盖，引擎版本锁不改变。
恢复适配的数值与资源释放资格仍须在修改后的代码身份上重新验证。

带有已生成输出的恢复 choice 已完成原始 prompt 的首次预填充。准入时通过
`Request.take_prefill_stats()` 消费目标新建请求的这项一次性统计状态，避免将
历史输出重算再次作为首次 prompt 预填充计量。目标缓存命中仍可覆盖完整历史，
不截断缓存命中、不扩展原 prompt 长度，也不伪造抢占次数。普通请求及尚无输出的
恢复请求保留原统计路径；请求预算、HTTP usage 和目标缓存命中指标仍分别验证。
CPU 回归须使用固定版本的真实 `PrefillStats`，执行 Scheduler 的首次预填充统计
分支，覆盖同步/异步调度和命中长度超过原 prompt 的情况；不能仅检查历史追加。

| 层 | 恢复责任 |
| --- | --- |
| frontend | 原 prompt、choice 映射、已提交输出、增量 detokenizer 的 token/text 偏移及 stop 缓冲、累计 logprobs、外部 request ID 与 usage；内部请求 ID 含迁移代次 |
| EngineCore/Scheduler | 原 prompt/output 分界、各 choice 的已生成 token、原有效 SamplingParams、已消耗预算、停止状态；KV 重算从未计算状态开始，保留既有 prefix-cache 查找与调度选取算法 |
| Worker/采样器 | 与 Core 一致的历史、真实 prompt 长度、输出计数、penalty/min_tokens 所需状态；不得用扩展后的 prompt 冒充原输入，不能只改 Python Request 而遗漏 batch/tensor 状态 |

恢复时引擎接收原有效 `max_tokens`/`min_tokens`，历史输出计入引擎计数；不能
在恢复完整历史后再传剩余 max_tokens，否则会双重扣减预算。只有保留原协议的
Transformers 分支继续由 Router 传入剩余预算。

随机请求保留参数语义及合法的随机采样状态；本项目不因此承诺与不迁移运行逐 token
恒等。多候选逐 choice 独立恢复，已经完成的候选不得再次采样，不能把 n 悄悄改为1。
新边界必须在冻结前检查所有待迁移请求是否能无损表达；无法表达则让源继续并回滚
目标，不能关闭源后再发现无法恢复。该保护也不代表功能最终完成。

## 内部信封与接口

信封为普通可序列化数据，不跨 Ray 保存引擎对象或 callable。字段至少包括：

- 批次的 `schema_version`、`epoch` 和 `requests`；源实例身份归 Router 所有权账本；
- 每份请求的 `schema_version`、`request_id`、模型标识、artifact/tokenizer 身份、
  原始 prompt IDs 与文本、原有效采样配置；
- 每个 `index` 的已提交生成 IDs、可见文本、累计 logprobs、自然完成原因、
  stop_reason，以及原 prompt logprobs；预算由原参数与 token 历史共同表达；
- Backend 保存代次决策与取消终态，拒绝冲突准备、重复激活和已取消请求复活。

固定 frontend 的 abort 输出会冲刷隐藏停止字符串缓冲。形成未完成快照前，
按原有效 buffer 长度撤回这部分可见文本，再由原 token 历史重建增量 detokenizer；
自然完成候选不裁剪。普通 DELTA 输出逐 choice 累计，恢复适配统一产生累计输出，
避免重复追加原历史。该接口不把内部 abort 作为用户完成原因。

Backend 内部接口沿用独立迁移协议，不加入所有后端必选的抽象方法：

1. `freeze_migration(epoch, request_ids=None)`：关闭该源的新准入，等待 Router 已发出
   的请求登记或进入终态，再等待相关生成迭代器停止，返回
   已提交的最终信封；保留模型、engine 和 GPU 预留，外部 generate 等待交接结果。
2. `prepare_migration(epoch, envelopes)`：目标验证身份及完整语义，完成最终同步；
   预热产生的 token 丢弃，用户生成保持未激活。返回与 request/choice 一一对应的确认。
3. `commit_migration(epoch)`：源的等待中 generate 返回内部交接信封，不返回用户
   响应。Router 已完成原子转交后，将同一外部请求送入目标
   `resume_migration(epoch, request_id)`；相同代次/请求只激活一次，重复调用等待
   同一结果。生成结果归原外部请求，不能产生第二个 HTTP 响应。
4. `rollback_migration(epoch)`：在源仍保留的 engine 上恢复原请求，重新开放准入；
   Router 已关闭时只清理，不重新启动用户已取消的请求。
5. `cancel_migration_request(epoch, request_id)`：对 prepared 与 active 状态都建立
   取消终态；迟到的 prepare/resume 不得复活它。

普通 generate 只在 MigrationRouter 传入内部 `migration_tracked=True` 时保留
完成登记，供已发出但尚未登记的 RPC 与已经完成的 RPC 区分。Router 在真实 RPC
结算并归还 slot 后调用 `forget_migration_request` 清理这些登记；不把本地 Future
取消误当成远端请求已终止。普通直调和内部预热不积累这份迁移账本。

自然完成的请求直接保留最终结果。预算为零或已达 EOS/stop 时不启动新的采样。
跨多次迁移始终用原预算与累计输出计数；prompt 的 usage 只计算一次。

## Router 所有权与提交顺序

Router 建立稳定 request ID 与 allocation Future 的关联。`request_count` 属于
整个外部请求，迁移不改变它；`allocated_instances[future]` 始终保存当前 slot owner。
实例/GPU reservation 与请求 slot 是两类账本，分别结算。

保留原迭代预热后，按以下顺序执行最终交接：

1. 对源建立独立的 admission/migrating 屏障。LB 不再分配源，autoscaler 不删除它；
   源仍计入实例数量，负向归还 slot 仍允许。处理已分配但尚未向源发起 generate 的请求，
   令其等待本代次提交或回滚，不能漏出最终快照。
2. 以 `_settle_owned` 观察 freeze/prepare RPC 的实际完成，即使调用者取消也不丢弃
   迟到副作用。冻结后、提交前失败，先恢复源，再关闭目标并归还其 reservation。
3. 在与 finalizer 共用的元数据锁下，更新 ready pool、migration_record、每个未完成
   请求的 slot owner，并一次性 source slot -1 / target slot +1。目标继承源容量。
   已经完成/取消的请求不再次转交；外部请求数不变。
4. 激活已经提交的目标请求；此后才退休源。目标已接管时，源清理失败不能撤销正常目标。
   finalizer 只 pop 最终 owner 一次。重复迁移通过显式 vLLM 协议循环处理。

不能用 `InstanceHandle.ready=False` 同时阻止负向 slot 归还，也不能依赖当前动态
`source.status=False` 字段作为 admission 屏障。Ray actor 必须为 freeze/abort/
rollback/shutdown 留出控制调用容量，避免所有 actor 并发位被长 generate 占满。
数据请求的准入容量仍由原 Router 配置限制，不因预留控制容量增加模型并发。

Transformers 继续走原信封与 chat template；缺省输出预算也要扣除源已完成 token。
不将 vLLM 新信封强加给 Transformers、Dummy 或抽象基类。

## Implementation 与 Validation

按可审查的边界分别实施并汇报，不能用某一层通过代替整个迁移完成：

1. 固定引擎恢复适配：对 Core、Worker、frontend 三层状态建立一致性断言；CPU
   检查 prompt/output 分界、各 choice 预算、penalties、min_tokens、EOS/stop、
   stop 字符串跨边界及 logprobs。构建身份和跨子进程插件发现必须独立验证。
2. Backend 冻结/恢复：覆盖首次输出前冻结、迭代器尾输出、自然完成、重复 epoch、
   多次迁移、源回滚及取消。不得由普通 shutdown 或用户取消触发自动续写。
3. Router 交接：复用 `test_migration_contract.py` / `test_migration_lifecycle.py`
   的事件与 reservation 账本，增加迟到分配、freeze/prepare 取消、提交时取消、
   finalizer 与转交锁竞争、目标失败、源清理失败和 autoscaler 并发。
4. 真实 Ray CPU 满载：全部数据准入位被长请求占据时，控制 RPC 仍能完成并清理；
   in-process RPC doubles 不能替代此检查。
5. 单独获得 GPU 批准后，验证同一个 HTTP 请求跨实例完成，记录最终源信封、目标
   恢复状态、完整输出 IDs/usage/finish_reason、cache 证据和源/目标资源释放。
   多候选和非默认采样组合分别准入，官方 Transformers 双模型路径单独回归。

TP2、更多模型和性能评测仍有各自关卡。CPU 通过不授予真实引擎或迁移资格。

### 同一 HTTP 请求的首批实机验收入口

`scripts/baseline/qualify_continuation.py` 为独立、显式批准的续写验收入口，
复用 D1 的私有 Store、双 worker Ray、HTTP 启动与 finally 清理；D1 默认入口、
检查集和历史结果含义不改变。新入口只消费 `stage-d2-continuation` 批次批准，
不能复用 `stage-d1-migration` 批准。此处 D2 是续写批次标识，不是 TP2 资格。

首批范围固定 Qwen3-0.6B、BF16、TP1、eager、V2、512 上下文和原 Store 产物。
每个场景通过真实 storage-aware allocation 触发迁移，并记录：

| 场景 | 原始请求参数与对照 |
| --- | --- |
| greedy | 单 choice、temperature=0、384 输出预算、logprobs=5 |
| seeded-multi | 两个 choices、temperature=0.7、seed=17、min_tokens=32、presence_penalty=0.2、frequency_penalty=0.1、384 输出预算、logprobs=5 |

两场景均显式 ignore_eos，避免短请求在目标冷加载前自然结束；持续流量使用新的
request ID，单个 ID 始终只发送一次 HTTP 请求，失败或取消不自动重发。资格必须
来自最终冻结信封中仍未完成的请求，且每个场景至少一个候选已在源生成并在目标
继续生成；仅迁移前观察 pending、零输出准入或迁移后新请求成功都不够。

在原 `trace_debug` 边界下，后端记录版本化 JSON 的 `frozen`、`prepared` 和
`resume` 事件，带实际 PID、epoch 和完整冻结信封。普通请求不打开该诊断。
验收器将源与目标 PID 关联到本批次 actor/cgroup，拒绝缺失、冲突或错配证据。
目标完整 history 和原 HTTP 响应必须保持冻结 token/text/logprobs 前缀，并与同参数
不迁移对照逐 choice 相等；所有请求预算、usage、终态与 slot/count 账本分别核对。
同批次零输出请求也校验完整结果，但不能冒充已生成请求接续。

对照响应与完整 history 在迁移开始前保存；冻结/准备信封、恢复事件、目标身份和
完整目标 history 在比较前落盘，不能等比较成功后再记录。比较失败须保留对应
request ID、choice index、首个不一致字段的路径及实际/对照值，并继续按原流程
清理；差异诊断不改变严格相等条件，也不将已冻结前缀与独立运行的数值差异自动
归因于迁移。CPU 回归覆盖源端已存在差异和恢复后出现差异两种失败路径。

严格对照另有一个迁移前必要条件：在源端已经观察到生成输出后、安装 allocation
probe 之前，读取本场景请求的累计 token/logprobs 前缀（包含观察期间刚完成的请求），
并与对应 choice 的对照前缀逐项比较。先保存读取结果，再保存比较结果；已有差异
必须在申请迁移分配前失败并清理流量，不能把这一差异归因于目标恢复。缺少有效输出
或 logprobs 也不能通过。此检查不比较尚在缓冲的文本，不替代迁移后的完整比较；
前缀相等只表示尚未发现差异，不证明不同批次或目标重算具有数值确定性。

首批的逐 choice 严格相等是该固定资格场景的额外要求，不扩大上文对一般随机请求的
功能承诺。不得通过删除 logprobs 比较、设置未经论证的容差或自动打开
`VLLM_BATCH_INVARIANT` 来取得通过。确定性计算配置需要作为独立变更审查，并在
固定版本、设备、attention backend 与 V2 runner 的实际组合上重新取得实机证据。

### 显式批次不变验收配置

D2 批准文件可显式选择 `numerical_profile: "batch-invariant-v1"`；缺省仍为
`default`。该选择只改变本批次验收的计算配置，不能推导普通默认服务具有相同的
数值确定性或性能。D1、Store 导出、已有产物与线上默认参数不随此选择改变。

选中该配置时，外部 runner 必须在启动 Python 前设置 `VLLM_BATCH_INVARIANT=1`。
D2 入口在任何模型导入或资源启动前检查批准字段与环境一致，拒绝未知配置、缺失
变量及 default 模式下意外继承的启用变量。批准仍须独立绑定 GPU、期限和部署身份；
配置字段本身不授权任何 GPU 使用。两个私有 Ray worker 节点显式传递同一变量。

模型配置固定 `attention_config.backend=FLASH_ATTN`，保留 TP1/BF16/eager、V2、
原请求参数、缓存及 Store 加载路径。仅此配置安装独立的只读 Worker 诊断扩展；
不修改 Store 扩展或上游 wheel。通过固定版本的命名 collective RPC 读取实际 Worker
的版本、PID/cgroup、GPU UUID/名称/计算能力、V2 runner 类、批次不变环境与已初始化
标记，以及 attention groups 和各层实现的启用状态。不能以 frontend 环境替代
实际模型进程证据，也不能由诊断方法初始化或打开批次不变模式。

每个场景的源端在发送对照请求前读取；目标端在取得身份后读取。结果先落盘再校验，
异常按已有 finally 路径清理。配置要求单个 TP1 Worker、A100/SM80、正确 GPU 与
cgroup、vLLM 0.31.0、V2 runner、实际已启用的批次不变实现、非空且全部为
FLASH_ATTN 的 attention layers。缺失或不匹配均失败，不回退其他 backend 或配置。
成功结果须包含两个场景各自的源和目标诊断通过记录。

CPU 回归覆盖批准/环境冲突、两个节点的显式传递、只读诊断、虚假启用标记、runner/
设备/attention 不匹配、失败证据和清理，以及 default/D1 不受影响。实际数学确定性
仍由独立批准的 GPU 批次证明：保留迁移前前缀检查及完整 tokens/text/logprobs/
cumulative_logprob 严格比较，不加容差，不删 seeded-multi 场景。

固定依据为 [v0.31.0 批次不变说明](https://docs.vllm.ai/en/v0.31.0/features/batch_invariance/)
及锁定 wheel 内 `batch_invariant.py` 的 SM80 实现、`gpu_worker.py` 的初始化调用、
V2 `model_runner.py` 的 attention groups 和 `flash_attn.py` 的每层启用标记。
`envs.py` 中仍写计算能力 9.0 的注释不能覆盖上述实现与文档；静态核对也不能当作
当前模型/设备/恢复链已经通过实机数值验收。

`continuation-result.json` 分开记录运行结果和外部释放证据；脚本不会自行预约 GPU，
也不会把进程内 cleanup 成功写成 GPU 已解锁。批准文件必须绑定精确 GPU UUID 顺序、
soft lock 动作、服务 cgroup、截止时间、不可变产物和匹配 commit/wheel 的部署验证。
短路径格式使用 `/tmp/sd2-<stamp>`、`gpu-stage-d2-<stamp>` 和
`sllm-d2-migration-<stamp>.service`。这些要求仍需由获批外部 runner 提供。

首批成功也不覆盖 EOS、跨边界 stop string、真实冻结后回滚/用户取消、多次迁移、
Transformers 双模型、其他模型、跨服务器或性能；这些组合继续作为独立验收项，
不能以两个场景通过标记阶段 D 或整个升级完成。CPU 流程替身只验证验收器的判断
和清理路径，不证明真实引擎执行结果。

## 固定版本接口依据

- [EngineCoreRequest](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/__init__.py)
- [Request](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/request.py)
- [Scheduler 及 streaming 更新](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/scheduler.py)
- [Worker 新请求数据](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/core/sched/output.py)
- [V1 GPUModelRunner](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/worker/gpu_model_runner.py)
- [Frontend OutputProcessor](https://github.com/vllm-project/vllm/blob/v0.31.0/vllm/v1/engine/output_processor.py)
