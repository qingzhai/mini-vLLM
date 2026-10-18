"""Load official AWQ tensors into the existing Qwen3 model, single GPU only."""
# 导入 gc 所需接口
import gc
# 导入 json 所需接口
import json
# 导入 Path 所需接口
from pathlib import Path
# 导入 torch 所需接口
import torch
# 导入 safe_open 所需接口
from safetensors import safe_open
# 导入 AWQLinear 所需接口
from awq_linear import AWQLinear


# 按需读取官方量化权重
class Reader:
    # 初始化当前对象的状态
    def __init__(self, directory):
        # 设置官方 Checkpoint 目录
        self.directory = Path(directory)
        # 设置实验配置
        config = json.loads((self.directory / "config.json").read_text())
        # 设置官方量化配置
        quant = config["quantization_config"]
        # 只接受官方 AWQ 四位权重量化格式
        assert quant["quant_method"] == "awq" and quant["bits"] == 4
        # 只支持 group128 与显式 Zero Point
        assert quant["group_size"] == 128 and quant["zero_point"] is True
        # 只解析 AutoAWQ GEMM 打包布局
        assert quant["version"].lower() == "gemm"
        # 设置实验配置
        self.config = config
        # 设置官方张量到分片文件的映射
        self.files = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        # 设置已消费张量名集合
        self.used = set()

    # 读取并登记一个官方权重张量
    def get(self, key):
        # Checkpoint 必须包含所请求的张量
        assert key in self.files, f"Missing checkpoint tensor: {key}"
        # 在指定资源或 CUDA 上下文内执行
        with safe_open(str(self.directory / self.files[key]), framework="pt", device="cpu") as reader:
            # 设置读取的张量
            tensor = reader.get_tensor(key)
        # 把当前元素加入集合
        self.used.add(key)
        # 返回读取的张量
        return tensor


# 沿输出通道合并量化投影
def merged(reader, prefixes, device):
    # 设置合并后的量化张量
    tensors = [torch.cat([reader.get(prefix + "." + suffix) for prefix in prefixes], dim=1).to(device)
               # 将前述构造应用到每个元素
               for suffix in ("qweight", "qzeros", "scales")]
    # 返回装载合并权重的量化 Linear
    return AWQLinear(*tensors)


# 将该调用设为无梯度推理
@torch.inference_mode()
# 用官方 AWQ 张量替换模型投影
def load_awq_model(model, directory):
    # 此适配器只面向已验证的 Qwen3 模型
    assert model.__class__.__name__ == "Qwen3ForCausalLM"
    # 设置按需加载权重的读取器
    reader = Reader(directory)
    # 设置 Target 层列表
    layers = model.model.layers
    # 模型层数必须与官方 Checkpoint 一致
    assert len(layers) == reader.config["num_hidden_layers"]
    # 设置目标设备
    device = next(model.parameters()).device
    # 确认当前并行配置在支持范围内
    assert torch.distributed.get_world_size() == 1
    # Release all dense projection placeholders before loading packed tensors.
    # 初始化待替换的 Linear 清单
    specs = []
    # 逐个处理行索引、layer
    for index, layer in enumerate(layers):
        # 设置当前层 Checkpoint 前缀
        prefix = f"model.layers.{index}"
        # 设置本层待替换投影分组
        groups = (
            # 合并 Q/K/V 的量化张量
            (layer.self_attn, "qkv_projection", [prefix + ".self_attn." + p for p in ("q_proj", "k_proj", "v_proj")]),
            # 替换 Attention 输出投影
            (layer.self_attn, "o_proj", [prefix + ".self_attn.o_proj"]),
            # 合并 FFN Gate/Up 的量化张量
            (layer.mlp, "gate_up", [prefix + ".mlp.gate_proj", prefix + ".mlp.up_proj"]),
            # 替换 FFN Down 输出投影
            (layer.mlp, "down_proj", [prefix + ".mlp.down_proj"]),
        # 结束当前调用或分组
        )
        # 逐个处理待替换模块的父对象、模块或张量名称、待合并的官方 Linear 前缀
        for parent, name, prefixes in groups:
            # 设置原始模块
            original = getattr(parent, name)
            # 当前替换路径没有实现量化偏置支持
            assert original.bias is None, "Bias support is not implemented in this adapter"
            # 设置原投影权重形状
            shape = tuple(original.weight.shape)
            # 向待替换的 Linear 清单追加本次内容
            specs.append((parent, name, prefixes, shape))
            # 先释放稠密占位权重，降低量化加载峰值
            setattr(parent, name, torch.nn.Identity())
            # 移除不再需要的原始模块
            del original
    # 回收旧稠密投影的 Python 引用
    gc.collect()
    # 释放已闲置的 PyTorch 显存缓存
    torch.cuda.empty_cache()
    # 逐个处理待替换模块的父对象、模块或张量名称、待合并的官方 Linear 前缀、原投影权重形状
    for parent, name, prefixes, shape in specs:
        # 设置新量化 Linear
        quantized = merged(reader, prefixes, device)
        # 检查张量形状与本轮数据一致
        assert (quantized.out_features, quantized.in_features) == shape
        # 用打包权重 Linear 替代原投影
        setattr(parent, name, quantized)
    # 逐个处理模块或张量名称、未量化参数
    for name, parameter in model.named_parameters():
        # 设置官方张量值
        value = reader.get(name)
        # 检查张量形状与本轮数据一致
        assert parameter.shape == value.shape, (name, parameter.shape, value.shape)
        # 未量化参数仍按原形状与模型 dtype 装入
        parameter.copy_(value.to(device=device, dtype=parameter.dtype))
    # 设置未被消费的 Checkpoint 张量
    unknown = set(reader.files) - reader.used
    # 未消费官方张量时拒绝宣布加载完成
    assert not unknown, f"Unconsumed checkpoint tensors: {sorted(unknown)[:10]}"
    # 每层应替换 QKV、O、Gate/Up、Down 四个融合模块
    assert len(specs) == 4 * len(layers)
    # 设置 model.awq_load_report
    model.awq_load_report = dict(quantized_fused_modules=len(specs),
                                 # 记录融合前对应的七个原始投影
                                 source_linear_layers=7 * len(layers),
                                 # 指定已消费权重张量数
                                 consumed_checkpoint_tensors=len(reader.used),
                                 # 指定未消费张量数
                                 unconsumed_checkpoint_tensors=0,
                                 # 指定量化格式
                                 quantization="AWQ W4A16 GEMM group128")
    # 输出本轮进度或诊断信息
    print("AWQ LOADER PASS:", model.awq_load_report, flush=True)
