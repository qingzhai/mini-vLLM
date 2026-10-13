"""Run FP16 and official AWQ checkpoint benchmarks in separate processes."""
from pathlib import Path
import os
import subprocess
import sys


def run_child(script, log_path, env):
    command = [sys.executable, "-u", str(script)]
    with log_path.open("x") as stream:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                stream.write(line)
                stream.flush()
                print(line, end="", flush=True)
            code = child.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def main():
    here = Path(__file__).resolve().parent
    run_dir = Path(os.environ["RUN_DIR"])
    run_dir.mkdir(parents=True, exist_ok=True)
    expected = [run_dir / name for name in (
        "awq-baseline-details.json", "awq-awq-details.json", "awq-comparison.json",
        "awq-baseline.log", "awq-awq.log", "awq-summary.log")]
    assert not any(path.exists() for path in expected), "Use a new RUN_DIR; existing results are preserved"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    env.setdefault("HF_HUB_OFFLINE", "1")
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for mode in ("baseline", "awq"):
        child_env = dict(env, QUANT_MODE=mode)
        run_child(here / "bench_awq.py", run_dir / f"awq-{mode}.log", child_env)
    run_child(here / "summarize.py", run_dir / "awq-summary.log", env)


if __name__ == "__main__":
    main()
