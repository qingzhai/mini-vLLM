# 来源与第三方依赖

本仓库是在开源推理项目之上的复现、扩展与实验整理。保留上游 Apache-2.0 LICENSE；新增代码同样按 Apache-2.0 发布。

- 基础引擎来自 [Wenyueh/MinivLLM](https://github.com/Wenyueh/MinivLLM)，初始检出提交 `00db3b477f33be1de91400483ae7d210c4d61868`。上游 README 说明其参考了 [Nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)。`src/myvllm` 含上游实现和本项目阶段性修改，不能把它们全部描述成从零独立实现。
- Qwen3 模型与 tokenizer 来自 [Qwen](https://huggingface.co/Qwen)。模型、tokenizer 和数据集遵循各自许可证，本仓库不分发权重。
- AWQ 使用官方 [Qwen3-8B-AWQ](https://huggingface.co/Qwen/Qwen3-8B-AWQ) checkpoint。量化 Linear 代码保留对 [AutoAWQ](https://github.com/casper-hansen/AutoAWQ) packing 格式的说明。此项目消费已有量化权重，没有自行执行 AWQ 校准。
- Draft 使用官方 [Qwen3-8B-DFlash-b16](https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16)，revision `9b41424b7109f9c5413454f481b09a82b85333f4`。官方 `modeling_dflash.py` 是运行时下载的外部依赖，不包含在发布源码中。项目工作包括 Target 适配、批量验证、接受与回滚、请求调度及 Graph 执行。
- `experiments/speculative_compare` 使用 [DeepSpec](https://github.com/deepseek-ai/DeepSpec) commit `005e03b81cec38b7da6399833d609ee89a2587f2`（MIT）及其发布的 Qwen3-8B EAGLE3、DFlash、DSpark checkpoint。官方源码在运行时下载，完整原始 ZIP 保留其 LICENSE、NOTICE 及源码，遵守其中 SpecForge 等第三方声明；Draft revision 和权重指纹在准备时锁定，不分发权重。输入从该固定提交的 GSM8K、HumanEval、MT-Bench 文件抽取，保留样本 ID、行号、输入 Token 和内容哈希用于复算；原数据与模型各自许可证仍适用。原 Mini-vLLM Target 串行适配已通过十二输入三轮 GPU 对照；官方提议/接受函数体保持原样，Target Cache 工厂由本项目适配。未训练 Draft，未实现 EAGLE3 树验证或三种 Draft 的普通 Serving/Graph 联合调度。
- ShareGPT 派生负载来自 [heka-ai/sharegpt-english-10k-vllm-serving-benchmark](https://huggingface.co/datasets/heka-ai/sharegpt-english-10k-vllm-serving-benchmark)，revision `854a6eb3c310f07f9e8b269a735bf6fe547e543d`。仓库保留样本 ID、行号和 token 哈希，原始对话和输入 token 数据由准备脚本在本地生成。`data/sharegpt-reference` 保留模型生成的参考输出 token 与验证元数据。

研究中的分阶段实现和结果来源见 `docs/PROJECT_SUMMARY.zh.md`、各 `results/*/SOURCES.json`。上游保留的 Llama 模块是基础引擎的导入依赖，本项目仅声明 Qwen 路径的运行验证。
