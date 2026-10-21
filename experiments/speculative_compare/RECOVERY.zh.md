# 恒源云迁移后的恢复

2026-10-20 用户提供的迁移后输出显示 `/hy-tmp` 为空，`/hy-netdisk` 没有挂载。当前实例没有旧工作目录，不能把备份镜像大小当作已备份数据盘的证据。平台说明迁移不会迁移 `/hy-tmp`，且会清除此目录：<https://gpushare.com/docs/instance/migration/>。

## 已完成的离线恢复

`minivllm-recovery-v1.zip` 从本地留存资料恢复了官方 DeepSpec Git checkout（原 commit、92 个文件原 SHA256）、十二条正式输入的 lock/workload、原始记录 ZIP 和 FP16 短对照入口。恢复过程遇到不同的已有文件会停止，不覆盖。

用户已执行第一阶段并返回：`RESTORED FILES: 123`、原 DeepSpec pin 正确；系统 `/usr/local/bin/python` 为 Python 3.11.12，Torch 2.9.1 / CUDA 12.8 / Triton 3.5.1 可复用，CUDA 可用。已创建继承系统包的 `venv-deepspec`，尚缺 Transformers 等 Python 依赖和全部模型权重。

## 第二阶段

`run_recovery_stage2.sh` 只补环境和权重，不启动 Benchmark。

- `restore_dependencies.sh` 用阿里云 PyPI 源解析原官方运行依赖，约束来自原三轮环境。排除直接安装 Torch/Triton；安装计划还会拒绝 Torch、Triton、TorchVision、TorchAudio 与 `nvidia-*` 替换，再把计划锁定为带 SHA256 的 wheel URL 安装。使用已有系统 GPU 软件栈，不重新下载安装 CUDA。
- `restore_model_weights.py` 用标准库和 curl，通过 ModelScope 官方文件 URL 下载 Qwen3-8B 与 EAGLE3 / DFlash / DSpark Draft。最多两个文件并发，保留 `.download` 断点；每 20 秒输出进度。只下载旧 lock 列出的必要文件，不下载另一种权重格式，不通过 SDK 再缓存一份模型。
- `config.json` 等已有归档元数据校验后直接恢复；Draft 的 `download_manifest.json` 按原脚本格式重建，三个文件的 SHA256 已核对与原件一致。该字段记载旧 HF commit，不把 ModelScope 的 `master` 冒充同名 commit。
- ModelScope `master` 只作传输来源，所有文件必须与原 SHA256 完全一致才能进入最终文件名；不一致立即停止，不刷新 lock 来放行。原 Target HF revision 未知，继续明确记为 `null`，身份由原文件哈希固定。
- 已验证文件重跑时跳过，已有不匹配文件保留；不删除失败样本、旧测量结果或修改阈值。权重恢复来源保存在 `recovery-environment/weights-restored.json`。

两个阶段使用现有 `/hy-tmp/minivllm-work`。模型未包含在恢复 ZIP 内；只有实际下载、哈希验证完成才能说服务器恢复了权重。第二阶段完成不代表 FP16 生成一致性通过；后续仍需执行独立的 `minivllm-speculative-fp16-v1/run_fp16_regression.sh`，不自动开始三轮正式实验。

## 运行与保存

将第二阶段包解压到工作目录，执行其中的 `run_recovery_stage2.sh`；用 `nohup` 将终端断开与任务生命周期分开。再次执行同一入口会复用已经恢复的文件。

本地保留的四份原始记录 ZIP、恢复包以及 GitHub 项目共同保存了代码与已返回实验记录，尚未返回的服务器独有文件不能据此宣称已恢复。下次迁移前应把需要保留的 `/hy-tmp` 内容上传至 OSS/下载至本地；保存镜像不能替代这一操作。
