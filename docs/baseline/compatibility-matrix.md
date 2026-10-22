# 升级资格与证据边界

本文件区分历史研究和当前集成资格。工程建立、依赖解析与 CPU 契约不执行
serving、CUDA 初始化或模型装载，不授予模型、迁移或性能资格。

| 对象 | 历史研究事实 | 当前 Fork 的新版资格 |
| --- | --- | --- |
| vLLM 0.31.0 + Qwen3-0.6B | BF16、TP1、eager、短输入原生/Store token 与记录 logprobs 一致 | 执行提交 fef30ee 的阶段 B 原生导出→独立禁网 native Store 恢复通过；339 个逻辑张量、2 条提示各 8 token/top-5 logprobs 精确一致；BF16/TP1/eager/512 上下文，阶段 C 执行提交 1478c0a 的限定 HTTP/Ray/Store 服务链也已通过 |
| Qwen3.5-9B 文本 | 四组短输入的各自引擎原生/Store 对照一致 | 未验证，不外推多模态、长上下文或迁移 |
| 后端请求生命周期 | 研究副本观察到 token 跟踪、stop 排空及 shutdown abort | 阶段 C 的 13 项真实运行检查通过：HTTP 服务链、4 个活动后端请求、取消、stop/shutdown、重启、Worker 故障和初始化失败后的显式清理重建；不代表自动故障迁移 |
| Store 原生 writer/reader | 曾有截断产物，后续研究使用 CPU 参考 writer | 原生 writer 完成与 reader 逻辑尾块对齐已修复；57 项 writer、25 项 reader CPU 测试、服务器真实 ext4 故障文件对照通过；两个扩展已重编译，真实 native Store 恢复通过；不以 CPU writer 替代原生路径 |
| 原版迁移调用链 | 固定 prompt 恢复调用返回，不是源活动快照接续 | D1 在54d099e通过真实双卡目标加载、源快照重放、缓存命中、路由切换、源释放及两类回滚；3个源活动请求终态为abort，仍无resume_generate/透明续写资格；等待阶段审查 |
| TP2、图、多模型、encode | 没有由阶段 0 增加的直接运行证据 | 未验证，按组合分别准入 |
| 安装与 ABI | 研究环境不是本 Fork 的完整依赖资格 | Linux 依赖已锁定；CPU HTTP→Ray actor 边界已验；Qwen3 准备链的 Python 3.12.13/Torch 2.13.0+cu130/vLLM 0.31.0/native Store ABI 已验；完整 serving 镜像尚未构建 |
| 配置与导入 | 旧包会在导入时强制载入其他后端 | 无执行依赖的配置投影与准备链真实引擎构造已验；限定 Qwen3 服务配置和生命周期已在阶段 C 验证 |

资格来自明确 commit、依赖、模型 revision、dtype、TP、上下文和执行方式的
直接测试。CPU 旧行为用例特意暴露继承缺陷，不能把其通过解释为该缺陷已修复。
不能因上游存在代码或历史探针通过而标记为当前 Fork 支持。
工程验证与原始研究归档来源见 [provenance](provenance.json) 和
[evidence-index](evidence-index.json)。

阶段 B 成功批次为 `20261022-082738`，执行代码固定为
`fef30ee9bab49fa7bf8b8b9ff53e60b9157ef4b3`；后续文档归档提交不改变该运行身份。
native wheel SHA256 为 `e73d1d16dc12698584a6a57bcf1800feaddc136bf72d12cdadd1d67cc78991e5`。
阶段 B/C 用户审查已通过；阶段 D/E 仍为独立关卡。

阶段 C 成功批次 `20261022-103635` 的执行提交为
`1478c0acd6d5f8ceca87d4cd1c8cafcb5bc8510a`，wheel SHA256 为
`3d83780758a9f91f0127bf746d24eab619a40fa4fdcd53ce44c74605d84b4569`。完整 13 项检查、故障边界和证据 SHA256
见 [运行证据](evidence-index.json#stage_c_runtime_qualification)。四路并发在直接 Ray backend 层观察；
HTTP 仍继承非流式返回和错误体使用 HTTP 200 的行为，不作吞吐或接口标准化声明。
