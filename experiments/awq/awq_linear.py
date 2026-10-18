"""Single-GPU AutoAWQ GEMM-format W4A16; independently implemented Triton GEMM.

Format reference: casper-hansen/AutoAWQ/awq/utils/packing_utils.py.
Eight output columns per int32, reverse nibble order [0,4,1,5,2,6,3,7].
Weights are unpacked inside each GEMM tile, never cached as a dense matrix.
"""
# 导入 torch 所需接口
import torch
# 导入 nn 所需接口
from torch import nn
# 导入 triton 所需接口
import triton
# 导入 triton.language as tl 所需接口
import triton.language as tl


# 编译为 Triton GPU 内核
@triton.jit
# 在 tile 内解包 INT4 并完成矩阵乘
def _gemm(A, W, Z, S, O, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          # 指定量化组长、tile 行列和归约宽度
          G: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # 设置 tile 的输出行
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    # 设置当前 tile 输出列
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    # 设置 tile 内归约索引
    inner = tl.arange(0, BK)
    # 每个 int32 打包八个输出通道的 INT4 值
    lane = cols % 8
    # 还原 AutoAWQ 交错 nibble 的位移
    shift = ((lane % 2) * 4 + lane // 2) * 4
    # 设置 FP32 累加器
    acc = tl.zeros((BM, BN), tl.float32)
    # 沿输入通道维逐块归约
    for start in range(tl.cdiv(K, BK)):
        # 设置当前输入通道索引
        kk = start * BK + inner
        # 设置 FP16 激活 tile
        a = tl.load(A + rows[:, None] * K + kk[None, :],
                    # 屏蔽超出输入边界的行与归约通道
                    (rows[:, None] < M) & (kk[None, :] < K), other=0)
        # 读取当前输入通道对应的打包 INT4 权重
        packed = tl.load(W + kk[:, None] * (N // 8) + cols[None, :] // 8,
                         # 屏蔽超出权重边界的输入与输出通道
                         (kk[:, None] < K) & (cols[None, :] < N), other=0)
        # 设置本组 Zero Point
        zero = tl.load(Z + (kk[:, None] // G) * (N // 8) + cols[None, :] // 8,
                       # 屏蔽超出 Zero Point 边界的输入与输出通道
                       (kk[:, None] < K) & (cols[None, :] < N), other=0)
        # 设置分组缩放系数
        scale = tl.load(S + (kk[:, None] // G) * N + cols[None, :],
                        # 屏蔽超出 Scale 边界的输入与输出通道
                        (kk[:, None] < K) & (cols[None, :] < N), other=0)
        # 取出对应权重的四位整数
        quant = (packed >> shift[None, :]) & 15
        # 按相同布局解出 Zero Point
        zero = (zero >> shift[None, :]) & 15
        # tile 内按 (q-z)×scale 反量化，不保留完整稠密权重
        dense = ((quant - zero).to(tl.float32) * scale.to(tl.float32)).to(tl.float16)
        # FP16 相乘并累积到 FP32
        acc = tl.dot(a, dense, acc)
    # 将累加结果写回输出矩阵
    tl.store(O + rows[:, None] * N + cols[None, :], acc,
             # 屏蔽超出输出矩阵边界的槽位
             (rows[:, None] < M) & (cols[None, :] < N))


# 保存打包权重并调用 W4A16 GEMM
class AWQLinear(nn.Module):
    # 初始化当前对象的状态
    def __init__(self, qweight, qzeros, scales, group_size=128):
        # 初始化当前对象的状态
        super().__init__()
        # 设置输入通道数
        self.in_features = qweight.shape[0]
        # 设置输出通道数
        self.out_features = qweight.shape[1] * 8
        # 设置每组量化通道数
        self.group_size = group_size
        # 确认数值类型符合该执行路径
        assert qweight.dtype == qzeros.dtype == torch.int32
        # 确认数值类型符合该执行路径
        assert scales.dtype in (torch.float16, torch.bfloat16)
        # 每个输入通道必须能完整划入量化组
        assert self.in_features % group_size == 0
        # 检查张量形状与本轮数据一致
        assert qzeros.shape == (self.in_features // group_size, self.out_features // 8)
        # 检查张量形状与本轮数据一致
        assert scales.shape == (self.in_features // group_size, self.out_features)
        # 把打包权重登记为随模型移动的非训练张量
        self.register_buffer("qweight", qweight.contiguous())
        # 保存每组打包的 Zero Point
        self.register_buffer("qzeros", qzeros.contiguous())
        # 保存每组缩放系数，保持 Checkpoint 原类型
        self.register_buffer("scales", scales.contiguous())

    # 执行当前模块前向
    def forward(self, x):
        # 当前 W4A16 内核要求 CUDA FP16 激活
        assert x.dtype == torch.float16 and x.is_cuda
        # 检查张量形状与本轮数据一致
        assert x.shape[-1] == self.in_features
        # 设置输出或迁移张量形状
        shape = x.shape[:-1] + (self.out_features,)
        # 设置当前特征
        x = x.reshape(-1, self.in_features).contiguous()
        # 设置输出张量
        out = torch.empty((x.shape[0], self.out_features), device=x.device, dtype=x.dtype)
        # 非空输入才启动量化 GEMM
        if x.shape[0]:
            # 设置 Triton 网格
            grid = (triton.cdiv(x.shape[0], 16), triton.cdiv(self.out_features, 64))
            # 按网格启动 tile 内解包与矩阵乘内核
            _gemm[grid](x, self.qweight, self.qzeros, self.scales, out,
                        # 传入实际矩阵 M、N、K
                        x.shape[0], self.out_features, self.in_features,
                        # 使用 group128 与 16×64×32 tile，配置四个 Warp
                        self.group_size, 16, 64, 32, num_warps=4)
        # 恢复输入的批次维度并替换为输出通道数
        return out.reshape(shape)


# 用 PyTorch 构造小算子的反量化参考
def dequantize_reference(layer):
    """Independent PyTorch unpacking; only used for the small operator check."""
    # 设置权重所在设备
    device = layer.qweight.device
    # 设置 AutoAWQ nibble 位移顺序
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=device, dtype=torch.int32)
    # 按 AutoAWQ 位序解出全部权重整数
    q = ((layer.qweight[:, :, None] >> shifts) & 15).reshape(layer.in_features, -1)
    # 按相同位序解出每组 Zero Point
    z = ((layer.qzeros[:, :, None] >> shifts) & 15).reshape(layer.in_features // layer.group_size, -1)
    # 将组级 Zero Point 扩展到各输入通道
    z = z.repeat_interleave(layer.group_size, dim=0)
    # 将组级 Scale 扩展到各输入通道
    scale = layer.scales.float().repeat_interleave(layer.group_size, dim=0)
    # 只为参考检查生成稠密 FP16 权重
    return ((q - z).float() * scale).half()
