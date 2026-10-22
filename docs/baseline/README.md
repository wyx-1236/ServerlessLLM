# ServerlessLLM-vLLM Baseline 工程入口

本目录保存独立 Baseline 的升级设计、来源和证据索引。目标为 vLLM 0.31.0；
阶段 B 的原生 Store 流水线与阶段 C 的限定服务链已实测并通过用户阶段审查；迁移、TP2 和模型/性能矩阵仍待后续阶段。

- [权威升级设计](engine-upgrade.md)：架构、兼容边界、阶段与验证要求。
- [仓库建立指南](repository-bootstrap.md)：Fork、独立 clone、固定上游及分支。
- [兼容范围](compatibility-matrix.md)：已验证事实与明确缺口。
- [来源清单](provenance.json)：原始提交、分支和候选引擎身份。
- [历史证据索引](evidence-index.json)：外部归档摘要及适用范围。
- [契约与补丁审计](patch-audit.md)：继承行为、缺陷与配置映射边界。
- [Linux 环境](environment.md)：依赖锁、CPU 检查及候选镜像配方。
- [服务生命周期](service-lifecycle.md)：后端/控制面所有权、关闭和手动限定验证入口。
- [迁移生命周期执行计划](migration-lifecycle-plan.md)：D1 原版调用链和资源回收的 scoped delta；不是已验证迁移资格。
- [Store 产物契约](store-artifact.md)：字节布局、逻辑张量视图和提交/复用校验。

## 仓库与部署身份

用户 Fork：[wyx-1236/ServerlessLLM](https://github.com/wyx-1236/ServerlessLLM)。
`origin` 指向用户 Fork，`upstream` 指向官方仓库。`codex/baseline-vllm` 是集成
分支，`codex/vllm-upgrade` 是升级分支，均从
`9f50241baa5386e06a9321c51f19a9ef5f964c2b` 起步。PR 向自己的集成分支提出，
不默认向官方提交。

当前提供无 GPU 的环境/契约检查入口；阶段 B 的 Qwen3-0.6B BF16/TP1/eager
原生导出与独立禁网 native Store 恢复已实测通过，执行身份和限制见
[兼容范围](compatibility-matrix.md)。阶段 C 在提交 `1478c0a` 上完成 13 项限定服务/生命周期检查；入口为
`scripts/baseline/qualify_service.py`，必须由获批的有界隔离 runner 执行，
具体前提见 [服务生命周期](service-lifecycle.md)。这不是最终 serving 镜像或通用部署资格；部署绑定确定 commit/tag。
模型权重、Store 产物、依赖缓存和大型结果保存在工程外，不纳入 Git。

D1 迁移候选已修复 Router/调度器的请求、actor 和预留所有权，并通过 356 项
CPU baseline 检查。提交 `54d099e` 已完成双卡原版迁移链与回滚的 13 项限定检查，
阶段报告等待用户审查；透明续写仍有缺口，整体验收未完成。手动入口为
`scripts/baseline/qualify_migration.py`。透明续写能力仍有缺口，详见
[迁移候选边界](service-lifecycle.md#d1-migration-candidate-bounded-checks-passed-review-pending)。

## 必须保留的执行关卡

每阶段完成后，在当前协作聊天中提交报告，等用户明确审查通过才开始下一阶段。
报告列明改动、产物/commit、验证及缺失、方向偏差、资源状态和下一阶段范围。
审查期间不启动下一阶段工作；修订后重新报告。

申请服务器 GPU 前必须报告实际服务器、全部物理 GPU 索引、hard/soft 锁模式、
用途、范围与时长，等待明确批准。包含导出转换、CUDA 导入探针和测试。
资源请求/释放分别使用 `hard lock GPU 0,1` / `hard unlock GPU 0,1` 格式，
实际索引逐一列出；soft 模式相应替换。结束或失败时先清理本任务进程和资源，
核验后报告释放状态，等待用户确认再继续任何项目工作。空闲卡不等于获批。

继承的 GPU workflow 不得通过标签、手工触发、runner 注册或后台作业绕过上述
审批。运行证据分别记录获批 GPU 批次；未通过 workflow 或 self-hosted runner 绕过审批。

## 与其他工程的边界

SLLM 后端、Store、测试和部署说明由此 Fork 拥有；CrossPool 的生产架构与
引擎迁移不属于本仓库改造。共同实验使用相同冻结 workload 和物理预算，
不要求合并运行环境。个人阶段报告另存研究知识库；不发布金山团队文档。
