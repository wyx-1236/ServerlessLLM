# Linux 环境冻结与无 GPU 验证

目标是 Linux x86_64、Python 3.12.13、vLLM 0.31.0、Torch 2.13.0 与
Transformers 5.17.0。`uv.lock` 限定 Linux x86_64 和 Python 3.12；所有传递依赖
由锁文件精确确定，不依据安装当日 latest 重新选择。uv 工具版本为 0.13.4。

## 来源与安装边界

根 `requirements.txt` / `requirements-worker.txt` 继续由 PEP 621 动态元数据读取，
保留主包名及 console script。Store 在自己的 pyproject 中提供静态依赖元数据，
以便不执行会导入 Torch/CUDA 编译逻辑的 setup.py 即可解析依赖。Store 的旧
requirements.txt 同步相同范围，PEP 621 是安装元数据的拥有者。

`tool.uv.sources` 强制 Store 来自本仓库的 `sllm_store`，不是 PyPI 的同名旧 wheel。
PyPI 0.8.0 元数据仍固定 Torch 2.7.0、Transformers 4.52.4，且旧 wheel 不能证明
现代 ABI 兼容。不得删除本地 source 或用 --no-sources 绕过这一边界。

原 FastAPI 0.115.12、OpenAI 1.52.0、Pydantic 2.11.5 与新版 vLLM 元数据不兼容；
原 Click 8.1.7 又与新版 Hugging Face Hub 冲突。相应范围已经更新，具体选中
版本以 uv.lock 为准。Ray 仍为 2.48.0，通过独立 Linux CPU 的 HTTP→Ray actor
测试才可认领该边界；这不等于完整 Controller、Store 或 serving 已通过。

完整环境包括 `worker` extra、`build` 组；CPU 检查使用 `cpu` 组的独立闭包。
原生构建工具组也锁定，Store build-system 的 Torch 同步为 2.13.0。
正式构建使用已锁工具和 --no-build-isolation，防止另装旧 Torch。

## 无 GPU 检查

在 Linux 上，以下命令只安装 CPU 契约所需依赖，不装 Torch/vLLM/Store：

```bash
export CUDA_VISIBLE_DEVICES=""
uv python install 3.12.13
uv export --frozen --only-group cpu -o /tmp/sllm-cpu.lock
uv venv /tmp/sllm-cpu
uv pip sync --python /tmp/sllm-cpu/bin/python --require-hashes /tmp/sllm-cpu.lock
/tmp/sllm-cpu/bin/python -m pytest tests/baseline -q
```

不附着已有 Ray 集群；CPU smoke 自建一个 num_gpus=0、num_cpus=1 的本地 Ray
实例，完成后 finally shutdown。测试检查没有导入 Torch。CLI 测试使用真实
Click 解析并替换服务操作，不能据此声称完整 SLLM Controller 已准入。

Windows 上可用单独 uv 管理的 Python 做纯配置/替身测试；未装 Linux 依赖的
测试应明确 skip，不把 Windows 通过当 Linux serving 资格。完整 serving 不在
Windows 安装。新的 Baseline CPU workflow 使用 GitHub hosted Ubuntu runner，
没有 --gpus、self-hosted、模型下载或服务 GPU 请求。

本地 WSL 跨会话使用时，检查环境和包缓存应放在自己的持久缓存目录；某些
WSL 配置会在发行版重启后清理 /tmp。Ray 的 socket 则使用短的私有临时路径，
避免把 pytest 的长用例名称加入 AF_UNIX 路径导致超过 107 字节。

## 镜像配方

`Dockerfile.baseline` 的 runtime-dependencies target 固定 CUDA 13.0.2 / Ubuntu
24.04 基础镜像和 uv 镜像摘要，安装锁定的 worker 与 build 依赖。该 target 明确
不构建/安装本地 Store 和 SLLM，也不应用旧 vLLM patch，不启动服务。

```bash
docker build --platform linux/amd64 -f Dockerfile.baseline \
  --target runtime-dependencies -t sllm-vllm-dependencies:0.31.0 .
```

基础镜像 manifest 已核对；镜像整体构建、系统 apt 包最终清单、原生扩展、
驱动/工具链和引擎实际导入仍需后续节点验证。不能把已核对基础 digest 的配方
称为已构建并冻结的最终镜像。原有 Dockerfile 是历史入口，不是新版部署入口。
阶段 B 验收 native Store 后再补全最终镜像及 wheel/ABI 清单；可能初始化 CUDA
的探针严格归入单独获批批次。

## 后续 GPU 输入

阶段 B 的 smoke 输入先固定 Qwen3-0.6B、BF16、TP1、eager、512 上下文与短
raw prompt。checkpoint revision 与历史锁定身份一致，执行前复核本地 snapshot
和 tokenizer/token IDs。该输入不是 GPU 批准，也不是已实现的自动导出启动器。
申请前核对具体服务器与物理 GPU 索引，报告逐卡锁模式和用途；批准后才能申请。

阶段 B 候选实现的准备接口及隔离条件见
[Store 产物契约](store-artifact.md#preparation-entry-and-ownership)。源快照可在 CPU
阶段通过 `freeze_source` 冻结并保存 JSON 凭据，再由准备入口读取与复核。完整
准备接口可能初始化 CUDA，不能把它作为无需批准的预检。

CPU 张量测试使用独立的 hash 锁 `tests/baseline/requirements-tensor-cpu.txt`，
包含 Torch CPU build；不修改 serving 的 `uv.lock`。安装和测试命令见
[测试入口](../../tests/baseline/README.md#cpu-torch-tensor-profile)。轻量 profile
继续要求没有 Torch/vLLM；CI 显式要求 g++，避免原生磁盘 writer 测试静默跳过。

## Bounded server qualification

The native Store pipeline (B) and the one-model service/lifecycle path (C) have
separate executable and wheel identities in [evidence-index.json](evidence-index.json).
Stage C used Ubuntu 22.04, Python 3.12.13, Torch 2.13.0+cu130, vLLM 0.31.0,
Ray 2.48.0, CUDA 13.0 and driver 610.57.04. Its wheel changes the Python loader
to accept an explicit Store root; both native extension binaries match the
qualified B wheel. This server environment is not a built final serving image.
