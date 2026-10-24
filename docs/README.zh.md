# 项目文档与阅读入口

项目围绕 Qwen3-8B，连接请求调度、分页 KV、投机验证、Target CUDA Graph 和多 GPU 执行。先建立请求与缓存的执行过程，再用正式实验解释收益和取舍。

| 你现在要做什么 | 打开哪份文档 | 应当得到什么 |
|---|---|---|
| 看懂代码与项目结构 | [源码学习指南](LEARNING_GUIDE.zh.md) | 普通执行链、实验适配点、函数顺序和手推练习 |
| 准备面试讲述 | [项目介绍与 32 题问答](INTERVIEW.zh.md) | 三种时长讲述、12 道优先题、可展开的源码与例子 |
| 查性能与正式结论 | [项目事实总结](PROJECT_SUMMARY.zh.md) | 实验索引、精度、比较对象、测量次数与局限 |
| 在自己的机器复现 | [复现说明](REPRODUCE.zh.md) | 环境选择、模型/数据准备、独立入口及归档方法 |
| 复现最新调度消融 | [ShareGPT 四组协议](SHAREGPT_FACTORIAL.zh.md) | Full / Chunk-only / Prefix-only / Chunk+Prefix 的独立开关与运行流程 |
| 使用项目简历 | [简历内容源](RESUME.zh.md) | 一段概述与五个技术点；Word 版从此文件导出 |
| 核对本次文档检查 | [验证记录](VALIDATION.json) | 本次本地检查范围、基准提交、GPU 证据链接与历史检查记录 |

第一次阅读按“学习指南 → 面试材料 → 项目总结”进行。只想查一个数字时直接进入 [results](../results/README.md)，无需租 GPU。复现时先选实验对应的环境，不把所有模型、Draft 和开关一起装进同一路径。

## 实验简称

| 简称 | 实验 | 主要比较对象 |
|---|---|---|
| S1 | 最新 ShareGPT 四组 | 同输入、同 Decode-first、Chunk × Prefix；每组三轮 |
| S0 | 历史 ShareGPT 三组 | 每组一次；Full 与 Chunk 的优先级不同 |
| G1 | DFlash 与 Target Graph | 相对 AR Graph；15 个候选＋Pending |
| D1 | EAGLE-3 / DFlash / DSpark | 相对配对 AR Eager；最多 7 个候选＋Pending，Batch=1 |
| M1 | Single / TP2 / DP2 / 1P1D | 单卡扩展收益，以及相同双卡资源的部署取舍 |
| Q1 | AWQ W4A16 | 权重存储、运行峰值显存和推理时间分别比较 |

这些简称用于定位记录，不代表执行开关或可以累乘的优化阶段。完整条件见[项目总结](PROJECT_SUMMARY.zh.md#experiment-index)。

## 简历与证据怎样同步

`RESUME.zh.md` 是简历的唯一文字来源，`RESUME.zh.docx` 是导出版。修改 Markdown 后，使用安装了 `python-docx` 的 Python 执行：

```bash
python scripts/export_resume_docx.py
```

脚本保留原 Word 的页面设置，统一标题、正文和重点数字格式，更新概述和五个技术点；输出后仍需打开 Word 或渲染检查分页。导出内容和源 Markdown 的一致性也可由脚本核验。

性能数字以 `results/` 原始摘要和 `SOURCES.json` 为准。正文可以精炼并突出贡献；模型来源、基线、输出工作量、统计口径及尚未通过项必须保留。已测得的倍数写成“达到基线的多少倍”，百分比增幅写成“提升多少百分比”，避免混淆。
