# EOS 与停止字符串的迁移实机验收增补

## 目标与审查边界

本增补定义阶段 D 内已批准实现与 CPU 检查的下一组验收入口；不改变服务端的生成、
迁移、调度或停止语义，不授予 GPU 权限。前置设计为
[请求恢复与交接方案](migration-continuation-engine-plan.md)。

验证原 HTTP 请求在源端已有输出、经过最终冻结和目标恢复后，由真实 EOS 或
跨冻结边界的停止字符串结束。停止发生在源端、预算耗尽、取消或内部 abort，
都不能替代这项证据。CPU 状态机检查与真实模型验收分别记录。

## 基线与方案选择

现有 `qualify_continuation.py` 固定 greedy 和 seeded-multi，使用
`ignore_eos=True`、384 输出预算。直接增加 `finish_reason=stop` 断言不足以证明
目标执行了停止判断；将请求截断或暂停引擎也会改变被测执行路径。

采用独立批准范围、复用原迁移流程的方案：以真实不迁移对照选择有界请求，
通过原 storage-aware allocation 触发迁移，以最终冻结信封判定资格。
不增加引擎暂停、强制 EOS、logits 修改或定点冻结钩子。
如果自然执行没有留下合格冻结请求，本场景未取得资格，不能降低条件或补造快照。

不选择仅检查终态的方案，因为它无法排除迁移前已经完成的请求；不选择
测试专用强制冻结方案，因为它需要额外改变运行路径并单独证明兼容性。

## 入口与兼容

- 新入口 `scripts/baseline/qualify_stopping.py --approval <json>`，只消费
  `scope=stage-d2-stopping`，运行路径前缀使用 `d2s`。D1、原 D2 批准不能互用。
- 首批只接受显式 `numerical_profile=batch-invariant-v1`，源和目标均验证实际
  Worker 的 V2、FLASH_ATTN、BF16/TP1/eager 与批次不变状态。普通服务默认值不变。
- 固定原 Qwen3-0.6B Store 产物、512 上下文、最多 128 prompt token、384 输出预算。
  不在线下载模型或覆盖 Store 产物。
- 复用私有 Store/Ray/HTTP、原请求发送、调度探针、迁移交接、账本结清、缓存检查
  与 finally 清理。用显式 suite/scenario 参数提取必要公共代码，不修改模块全局
  `SCENARIOS`，不复制一份完整生命周期，也不改变原入口的参数与通过条件。
- 新证据文件为 `stopping-result.json`，包含 suite、批准身份、checks、cases、
  scenarios、实际数值配置、cleanup 和限制。外部释放证据单独保存。

## 固定场景

### 自然 EOS

场景名 `eos`；temperature=0、n=1、min_tokens=32、max_tokens=384、logprobs=5、
ignore_eos=False，不设置用户 stop/stop_token_ids。使用固定 tokenizer 的 chat
template，`enable_thinking=False`，按以下顺序最多探测三个提示，选第一个合格者：

1. `List the integers from 1 to 20, separated by commas, then finish.`
2. `Write a short paragraph of about 60 words explaining how a computer stores data.`
3. `Give exactly five short numbered tips for organizing files on a computer.`

每个提示的实际字符串、template 后 prompt IDs、有效 EOS IDs、响应和 history
都先落盘。prompt 超过 128 token 或不迁移对照未在 32 至 383 个输出 token 内
以 EOS 结束，该提示不合格；三个均不合格则失败，不无限搜索或替换停止规则。
选定后再发一个独立同参数对照，严格一致才进入迁移流量。

有效 EOS IDs 来自同一已验证 tokenizer/model generation 配置，保留来源；实际
冻结 SamplingParams 必须与这些配置及请求一致。每个合格冻结 choice 必须未完成、
已有输出且尚未包含 EOS。目标最终比冻结历史更长，最后一个 token 属于有效 EOS
集合，finish_reason=stop，stop_reason 为主 EOS 的空值或相应附加 EOS token ID，
输出未耗尽预算。完整结果仍与独立对照逐项严格相等。

### 跨冻结边界停止字符串

场景名 `stop-string-exclude`、`stop-string-include`，分别设置
include_stop_str_in_output=False/True。其余参数为 temperature=0、n=1、
min_tokens=0、max_tokens=384、logprobs=5、ignore_eos=True。

沿用原 D2 的 96-token 提示，先取得无 stop 的完整 384-token 探测历史。使用同一
tokenizer 及有效解码选项分别解码前 32 和前 320 个输出 token，要求二者均为
完整探测文本的严格前缀。以两者之间的非空字符串作为唯一 stop；记录 token
边界、字符边界、字符串、解码选项和探测历史。无法得到可靠前缀时失败，不猜测
token 与字符的对应关系。长字符串用于扩大可观察的跨边界区间，不代表性能基准。

字符串内容仍由上述固定 token 切片决定；记录选取字符区间与该字符串的真实首次
匹配区间。重复输出可能使首次匹配早于选取位置，此时使用实际首次匹配位置作为
`stop_start/stop_end`，不得把后面的重复匹配当成实际终止位置。

对每个 include 选项单独取得真实不迁移对照。要求第一次 stop 匹配恰在记录的首次
匹配区间、自然以该字符串结束且未耗尽预算；include/exclude 文本分别遵循上游
语义。只有该验证通过，才使用完全相同的请求体启动持续 HTTP 流量。

对最终冻结的未完成 choice，重建冻结 token 的未裁剪文本并核对它是探测文本
前缀。资格条件为 `stop_start < frozen_text_length < stop_end`：匹配字符串的一部分
在源端已经生成，剩余部分必须由目标生成。最终 token 数必须增加，finish_reason
为 stop、stop_reason 精确等于选定字符串；保留冻结可见文本与 logprobs，并与同
include 选项的独立对照严格一致。只发生在源端或完全位于目标的 stop 不提供跨边界资格。

## 通过与失败判定

三场景必须分别有至少一个合格的已生成请求完成目标续写。所有最终交接请求都必须
通过现有通用信封、完整 tokens/text/logprobs/cumulative_logprob、usage、参数预算
与 HTTP 单次完成校验；其他正常完成的请求不能冒充停止边界资格。

场景专用校验返回 `qualified_request_ids` 与逐 choice 边界记录，不能只保存布尔值。
汇总器重新检查场景集合、合格请求归属、六份源/目标 Worker 报告、通用 checks 和
cleanup，不因旧 D2 的 passed 字段或仅有 stop 终态而通过。

保留迁移前前缀比较，数值不一致在申请迁移分配前失败。探测、对照、冻结、恢复和
比较输入在断言前保存；不符合边界时记录具体条件。每个 HTTP ID 只发送一次，不
自动重试原请求。整批失败或超时仍排空本批次任务、关闭私有服务并由外部 runner
验证资源释放；不能因校验异常跳过清理。

外部归档必须显式包含完整 result、批准、release 和源/目标事件，生成文件哈希
清单并验证下载。文件大于常规日志上限时写入补充压缩包，禁止静默遗漏必需证据。

## 实施与 CPU 验证

1. 增加纯 CPU 的场景构造和边界验证模块 `stopping_evidence.py`，先覆盖反例：
   假 EOS、用户 stop token 冒充 EOS、预算终态、源端已结束、零输出冻结、目标
   没有继续、stop 完全在源/目标、错误 include 文本、重复候选、数值差异及证据缺失。
2. 把原 D2 流程按需提取为显式参数化的公共执行函数，接入新入口。原 greedy/
   seeded-multi 请求体、批准、检查集合及清理回归必须保持。新增 CPU HTTP/actor
   流程替身验证三个场景成功、探测耗尽、边界失败、异常和取消清理。
3. 不新增异步 `__ray_call__` 回调；如需要新 actor 方法，先用真实零 GPU Ray 验证
   成功与异常传递。新增模块导入不得导入 Torch/vLLM 或初始化 CUDA。
4. 完成完整 CPU 回归、依赖跳过补充核对、独立服务器无 GPU 部署及入口拒绝检查。
   再提交具体服务器/GPU、部署身份、范围和时长供单独授权。

## 范围外

本批次不覆盖随机多 choice 的停止组合、真实冻结后故障回滚/取消、同请求反复迁移、
Transformers GPU、TP2、其他模型或性能；不能据此关闭阶段 D 或整个升级。
不以 CPU 32 项状态机测试或此前两个 D2 场景的 GPU 成功代替本批次实测。
