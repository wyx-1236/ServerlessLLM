# 原版契约与历史补丁审计

参照为 `9f50241baa5386e06a9321c51f19a9ef5f964c2b`。历史 candidate02 累计补丁
SHA256 为 `0c8302d23a523ee8406f0d624cd0a08b1e8b1079d8481dd232f80e11b212e0e7`。
该补丁在原版干净 checkout 上 `git apply --check` 通过；未整体应用到当前分支。
它在已修改的后端导入文件上不能直接套用，按以下语义逐项迁入。

## 历史补丁的取舍

| 文件/改动 | 决定 | 理由与后续归属 |
| --- | --- | --- |
| `sllm/backends/__init__.py`：延迟导入 vLLM | 采用其目的并扩展到三个后端 | 配置与控制面导入不应强制加载另一后端；公开类名保留，访问时才加载具体实现 |
| `transformers_backend.py`：异常后清理生成状态 | 不直接迁入 vLLM | 原版 Transformers 同步 InferenceStatus 与 vLLM 异步 trace 不同；相应异步 finally 清理归阶段 C |
| 同文件：thinking 透传与模板失败策略 | 不迁入 vLLM | 不改变原版 vLLM 的 messages 拼接规则；评测驱动显式冻结输入 |
| 同文件：目标缓存少一 token 的恢复修复、resume_generate | 不复制 | 依赖 HF past_key_values 结构；vLLM 保持原有 token 重放机制，阶段 D 单独验收接续缺口 |
| `migration_router.py`：异步等待、计数 finally、失败回滚/退休 | 保留为阶段 C/D 必需修复依据 | 是共享控制面候选修复，但含实际迁移流程改动，不能在阶段 A 未验证时批量应用 |
| `sllm_store/client.py`：显式地址优先，否则 SLLM_STORE_ADDRESS | 保留为阶段 B Store 接入候选 | 默认仍为 127.0.0.1:8073；须连同调用点和 IPC 生命周期验证 |
| `sllm_store/torch.py`：使用客户端默认解析 | 随上述地址修复一起处理 | 避免确认加载与装载访问不同服务；本阶段未修改 Store 装载逻辑 |
| `sllm_store/transformers.py`、`utils.py`：同类地址替换 | 仅共享地址契约回归时采用 | 不因升级 vLLM 顺带改变 HF 量化/LoRA 行为 |
| `transformers_backend_test.py`：恢复/路由测试 | 保留其边界思路，不复制为 vLLM 资格 | 必须用 vLLM 的真实接口与独立预期测试 |

## vLLM 研究原型的取舍

Counter 替换和 task 构造参数变化归阶段 C 的引擎适配；阶段 A 先提供无引擎导入的
配置投影。Worker 扩展 + 字符串 collective RPC、parameter/buffer 绑定及 dtype 保留
归阶段 B。原型 `rank_0`、绝对路径、CPU 参考 writer、整对象 del 回收均不能作为
正式实现直接采用；不得打开 pickle callback 回退或修改上游模型执行源码。

## 原版实际行为矩阵

`tests/baseline/test_legacy_contract.py` 运行当前仍未改动的 `vllm_backend.py` 和
`model_downloader.py`，仅替换无法在无 GPU 环境运行的引擎/依赖。测试不加载模型，
不证明新版真实引擎兼容。以下“缺陷”是测试观察到的继承行为，不是目标契约。

| 边界 | 原版观察 | 目标/验收归属 |
| --- | --- | --- |
| prompt | input_tokens 优先于 prompt；messages 使用 role: content 按行拼接 | 保持该优先级，模板规则不暗改 |
| 输出与 usage | ID 与逻辑模型名保留，prompt/output 数量累加，返回完整非流式响应 | 阶段 C 对真实 HTTP 与有效采样参数验收 |
| 采样非法 | 返回 Invalid sampling parameters 错误，不启动生成 | 保留可定位错误 |
| 多请求 | CPU 替身下 ID/采样预算相互独立 | 不外推 continuous batching 或真实并发资格 |
| 活动登记 | 首次输出前计数为 0 | 缺陷：已接纳请求必须立即登记，阶段 C 修复 |
| 生成异常 | 首个输出后异常会残留 trace | 缺陷：finally 清理，阶段 C 修复 |
| trace_debug | 完成历史仍算活动项，会影响 stop 排空 | 缺陷：历史与活动集合分开，阶段 C 修复 |
| stop | 对已经观察到输出的活动请求拒绝新工作并等待完成 | 保留 drain 语义，同时补首输出前竞态 |
| shutdown | 删除 engine 引用，没有调用真实关闭方法 | 缺陷：阶段 C 证明 Worker/EngineCore 实际退出 |
| 缓存重放 | 对给定 token 序列执行 max_tokens=1 | 不等于 KV 命中或原请求接续，阶段 D 验证 |
| 接续 | vLLM 没有 Router 所调用的 resume_generate | 原版缺口，不能伪称透明迁移已实现 |
| 导出完整性 | 只要目标目录存在就直接返回，即使空目录 | 缺陷：阶段 B 增加清单、校验与原子提交 |
| 导出异常 | writer 抛错包装为 RuntimeError，移除本次部分目录 | 保留错误传播，阶段 B 增加已发布产物保护和多 rank 事务 |

## 配置映射契约

`sllm/backends/vllm_config.py` 只做投影，不启动引擎；待阶段 C 显式接入后端构造。
调用者传入真实 AsyncEngineArgs 的字段集合和版本，唯一接受的版本为 0.31.0。
CPU 测试字段 fixture 从固定 v0.31.0 的官方 arg_utils.py 用 AST 提取，来源和
SHA256 在 fixture 旁保存；它不替代真实构造器的类型/组合校验。

| 原字段 | 新处理 |
| --- | --- |
| torch_dtype | 映射到 dtype；两者不同时报错 |
| task=auto | 不传已经移除的 task；保留新版 auto runner/convert 默认 |
| task=generate | runner=generate |
| task=embed / embedding | runner=pooling，convert=embed |
| task=classify | runner=pooling，convert=classify |
| 其他旧 task | 明确拒绝，不静默当生成；须另做 runner/pooling 组合适配与模型资格 |
| trace_debug、hf_model_class | SLLM 字段，不泄漏给 AsyncEngineArgs |
| device_map=auto | 接受共享配置默认值，不伪称 HF placement 在 vLLM 生效；其他值拒绝 |
| 未提供 load_format | 模型路径为 STORAGE_PATH/vllm/model，格式为 serverless_llm |
| 显式 load_format | 必须提供 pretrained_model_name_or_path；仅用于明确配置的加载路径/对照 |
| eager / prefix cache | 保留原默认 False / True；显式值不覆盖 |
| executor | 本机 mp 为候选默认，TP/设备绑定仍待实测 |
| 未知拼写或不支持字段 | 明确报错，禁止静默过滤 |

此映射不声明任一池化模型、非默认 task 或多 GPU 已通过准入。未实现的旧 task
仍是正式完整兼容声明的限制，不能在性能报告中省略。
