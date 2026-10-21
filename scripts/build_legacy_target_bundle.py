#!/usr/bin/env python3
"""Package an offline runner; no installations, downloads or GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'experiments/speculative_compare'
PREFIX = 'minivllm-legacy-target-v1'


def build(destination, formal=False):
    prefix = 'minivllm-legacy-target-formal-v1' if formal else PREFIX
    manifest = json.loads((HERE/'legacy-source-manifest.json').read_text())
    files = {}
    for name in ('benchmark.py', 'protocol.py', 'precision.py', 'legacy_target.py',
                 'legacy-source-manifest.json', 'prepare_fp16_regression.py',
                 'analyze_divergence.py', 'run_legacy_target_regression.sh',
                 'verify_legacy_target_cpu.py'):
        files[name] = (HERE/name).read_bytes()
    if formal:
        for name in ('prepare_legacy_formal.py', 'run_legacy_target_formal.sh'):
            files[name] = (HERE/name).read_bytes()
    files['formal-workload.json'] = (ROOT/'results/speculative_compare/formal-20261019/workload.json').read_bytes()
    for name, digest in manifest['sha256'].items():
        content = (ROOT/name).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError('Archived Target source changed: ' + name)
        files['legacy-source/'+name] = content
    files['README.md'] = '''# 原 Mini-vLLM Target + 官方 DeepSpec Draft：离线短回归

复用现有 Python 环境、四个模型和原十二条输入 lock，不安装、不下载。
Target 为原 Mini-vLLM FP16 模型、真实 FP32 Head、固定形状算术和逐 Query PagedAttention；Draft 为当前官方 BF16 EAGLE3/DFlash/DSpark。
只做三条原输入、256 输出、一轮，全部七种模式及配对 AR。没有 CUDA Graph，也不是普通 Serving。
Target Top-2 收集已开启，计时仅为诊断；包生成时尚未 GPU 验收。
原 HF FP16 回归为 20/21 组投机请求一致，Confidence 0.8 的 mt-bench:9 在索引24分叉。本包不改阈值或降低判定标准。

在现有服务器解压后执行：

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
bash "$WORK_ROOT/minivllm-legacy-target-v1/run_legacy_target_regression.sh"
```

结果写入新的 results/speculative-legacy-target-regression-* 目录。
下载指针：$WORK_ROOT/speculative-legacy-target-latest-records.zip。
QUALIFIED GREEDY COMPARISON: True 只说明本短回归通过，不代表完整正式实验通过。
manifest 锁定原 Target 源码；package-manifest.json 锁定包内文件。
'''.encode()
    if formal:
        files['README.md'] = '''# 原 Mini-vLLM Target + 官方 Draft：正式测量

已经通过的短回归为三条请求、七种投机模式，共21/21组逐Token一致。
本包复用相同Target计算源码和原四个模型。唯一测量配置变化是恢复原十二条输入、三轮，关闭Top-2收集；每条256输出，忽略EOS。
所有模式与配对AR在同一套Target算术下运行；没有CUDA Graph或并发Serving，不使用历史HF结果作为本轮基线。
仅复用现有环境和本地权重，不安装、不下载、不追加Smoke。完整正确性仍待本轮结果，不能提前声称通过。

```bash
export WORK_ROOT=/hy-tmp/minivllm-work
bash "$WORK_ROOT/minivllm-legacy-target-formal-v1/run_legacy_target_formal.sh"
```

入口读取 results/speculative-legacy-target-latest.txt 指向的通过短回归，核对原记录和计算源码，然后创建新目录。
若该指针已变化，可设置 REGRESSION_DIR=/hy-tmp/minivllm-work/results/实际通过目录。
新结果在 results/speculative-legacy-target-formal-*，下载指针为 speculative-legacy-target-formal-latest-records.zip。
保留全部输出Token与接受轨迹；分叉不会通过删样本或修改阈值掩盖。
'''.encode()
    package = dict(purpose='Offline formal runner based on passed short regression' if formal else 'Offline diagnostic runner; no GPU correctness claim',
                   sha256={name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())})
    files['package-manifest.json'] = (json.dumps(package, ensure_ascii=False, indent=2)+'\n').encode()
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(prefix+'/'+name, data)
    print('BUNDLE:', destination)
    print('FILES:', len(files))
    print('BYTES:', destination.stat().st_size)
    print('SHA256:', hashlib.sha256(destination.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--formal', action='store_true')
    args = parser.parse_args()
    prefix = 'minivllm-legacy-target-formal-v1' if args.formal else PREFIX
    build(args.output or ROOT/(prefix+'.zip'), formal=args.formal)
