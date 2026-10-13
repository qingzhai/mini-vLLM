# DFlash + Target CUDA Graph：精选结果

单 A100-PCIE-40GB，Qwen3-8B FP16 主干 + FP32 LM Head，官方 BF16 Draft。每请求生成 256 tokens；两种固定提示词负载，各四种模式测量三轮。`summary.csv` 保留三轮中位数，JSON 另记录条件、校验范围和计时范围。

相对 **AR Graph** 基线，联合模式单请求为 **2.987×**（5.513→1.846 s），batch=2 的三请求为 **1.523×**（11.653→7.651 s）。三请求吞吐为三个请求合计输出 768 tokens 的总吞吐。

24 次运行、48 次请求执行、12,288 个输出 token 全部匹配；8 组同历史前向精确一致，Graph fallback 为 0，KV 页全部归还。

本轮不是 ShareGPT 压测；Graph 只覆盖 Target 前向。Draft、初始 Prefill、接受判断和回滚在图外，所有模式驻留相同 Draft 和图资源。完整原始记录本地保存，`SOURCES.json` 提供 ZIP 与原始记录 SHA256。
