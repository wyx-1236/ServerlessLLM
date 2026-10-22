# 升级资格与证据边界

本文件区分历史研究和当前集成资格。工程建立、依赖解析与 CPU 契约不执行
serving、CUDA 初始化或模型装载，不授予模型、迁移或性能资格。

| 对象 | 历史研究事实 | 当前 Fork 的新版资格 |
| --- | --- | --- |
| vLLM 0.31.0 + Qwen3-0.6B | BF16、TP1、eager、短输入原生/Store token 与记录 logprobs 一致 | 阶段 B 准备流水线已有实现和 CPU 契约；真实原生/Store 恢复尚未验证 |
| Qwen3.5-9B 文本 | 四组短输入的各自引擎原生/Store 对照一致 | 未验证，不外推多模态、长上下文或迁移 |
| 后端请求生命周期 | 研究副本观察到 token 跟踪、stop 排空及 shutdown abort | 未验证 Controller/Ray/HTTP 及异常回收 |
| Store 原生 writer | 曾有截断产物，后续研究使用 CPU 参考 writer | 尾块写入/错误传播已修复，独立 Linux O_DIRECT 故障测试通过；完整扩展和服务器文件系统仍未验证，是发布阻断项 |
| 原版迁移调用链 | 固定 prompt 恢复调用返回，不是源活动快照接续 | 未验证；原版 vLLM 缺少 Router 接续分支所需 resume_generate |
| TP2、图、并发、多模型、encode | 没有由阶段 0 增加的直接运行证据 | 未验证，按组合分别准入 |
| 安装与 ABI | 研究环境不是本 Fork 的完整依赖资格 | Linux 依赖已锁定；真实 CPU HTTP→Ray actor 边界已验，Store ABI/完整镜像仍未验证 |
| 配置与导入 | 旧包会在导入时强制载入其他后端 | 无执行依赖的配置投影已验；尚未接入真实引擎构造 |

资格来自明确 commit、依赖、模型 revision、dtype、TP、上下文和执行方式的
直接测试。CPU 旧行为用例特意暴露继承缺陷，不能把其通过解释为该缺陷已修复。
不能因上游存在代码或历史探针通过而标记为当前 Fork 支持。
工程验证与原始研究归档来源见 [provenance](provenance.json) 和
[evidence-index](evidence-index.json)。
