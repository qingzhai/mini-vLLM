# AWQ W4A16：精选结果

消费官方 Qwen3-8B-AWQ checkpoint（固定 revision），实现量化 Linear 推理与解包参考检查。单 A100 40GB，FP16 激活 + 额外 FP32 LM Head，Graph/Prefix/Chunk 关闭；四种短长输入和 batch=1/4 场景，各测三轮，指标为各自中位数。

权重及量化元数据：**16,381,470,720→6,098,479,104 bytes（−62.772%）**。包含本项目额外 FP32 LM Head：**18,870,790,144→8,587,798,528 bytes（−54.492%）**。所测峰值 allocated 显存约从 19.87～20.03 GiB 降至 10.31～10.47 GiB。

当前实现吞吐低于 FP16：short-b1 12.111→3.429 tokens/s；其他场景见 CSV。因此本轮结论是权重存储和显存下降，没有量化加速结论。

903 个 checkpoint 张量全部消费，6 项量化 Linear 参考检查通过；另验证有限值生成和页回收。未自行完成 AWQ 校准，也未执行任务质量评测。完整原始记录本地保存，`SOURCES.json` 提供 SHA256。
