"""Load official AWQ tensors into the existing Qwen3 model, single GPU only."""
import gc
import json
from pathlib import Path
import torch
from safetensors import safe_open
from awq_linear import AWQLinear


class Reader:
    def __init__(self, directory):
        self.directory = Path(directory)
        config = json.loads((self.directory / "config.json").read_text())
        quant = config["quantization_config"]
        assert quant["quant_method"] == "awq" and quant["bits"] == 4
        assert quant["group_size"] == 128 and quant["zero_point"] is True
        assert quant["version"].lower() == "gemm"
        self.config = config
        self.files = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.used = set()

    def get(self, key):
        assert key in self.files, f"Missing checkpoint tensor: {key}"
        with safe_open(str(self.directory / self.files[key]), framework="pt", device="cpu") as reader:
            tensor = reader.get_tensor(key)
        self.used.add(key)
        return tensor


def merged(reader, prefixes, device):
    tensors = [torch.cat([reader.get(prefix + "." + suffix) for prefix in prefixes], dim=1).to(device)
               for suffix in ("qweight", "qzeros", "scales")]
    return AWQLinear(*tensors)


@torch.inference_mode()
def load_awq_model(model, directory):
    assert model.__class__.__name__ == "Qwen3ForCausalLM"
    reader = Reader(directory)
    layers = model.model.layers
    assert len(layers) == reader.config["num_hidden_layers"]
    device = next(model.parameters()).device
    assert torch.distributed.get_world_size() == 1
    # Release all dense projection placeholders before loading packed tensors.
    specs = []
    for index, layer in enumerate(layers):
        prefix = f"model.layers.{index}"
        groups = (
            (layer.self_attn, "qkv_projection", [prefix + ".self_attn." + p for p in ("q_proj", "k_proj", "v_proj")]),
            (layer.self_attn, "o_proj", [prefix + ".self_attn.o_proj"]),
            (layer.mlp, "gate_up", [prefix + ".mlp.gate_proj", prefix + ".mlp.up_proj"]),
            (layer.mlp, "down_proj", [prefix + ".mlp.down_proj"]),
        )
        for parent, name, prefixes in groups:
            original = getattr(parent, name)
            assert original.bias is None, "Bias support is not implemented in this adapter"
            shape = tuple(original.weight.shape)
            specs.append((parent, name, prefixes, shape))
            setattr(parent, name, torch.nn.Identity())
            del original
    gc.collect()
    torch.cuda.empty_cache()
    for parent, name, prefixes, shape in specs:
        quantized = merged(reader, prefixes, device)
        assert (quantized.out_features, quantized.in_features) == shape
        setattr(parent, name, quantized)
    for name, parameter in model.named_parameters():
        value = reader.get(name)
        assert parameter.shape == value.shape, (name, parameter.shape, value.shape)
        parameter.copy_(value.to(device=device, dtype=parameter.dtype))
    unknown = set(reader.files) - reader.used
    assert not unknown, f"Unconsumed checkpoint tensors: {sorted(unknown)[:10]}"
    assert len(specs) == 4 * len(layers)
    model.awq_load_report = dict(quantized_fused_modules=len(specs),
                                 source_linear_layers=7 * len(layers),
                                 consumed_checkpoint_tensors=len(reader.used),
                                 unconsumed_checkpoint_tensors=0,
                                 quantization="AWQ W4A16 GEMM group128")
    print("AWQ LOADER PASS:", model.awq_load_report, flush=True)
