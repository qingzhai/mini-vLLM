import atexit
import torch.distributed as dist
import time
import torch.multiprocessing as mp

from myvllm.engine.sequence import Sequence
from myvllm.engine.scheduler import Scheduler
from myvllm.engine.model_runner import ModelRunner
from myvllm.sampling_parameters import SamplingParams
from transformers import AutoTokenizer


def worker_process(config, rank, event):
    """Worker process function that initializes ModelRunner and enters loop."""
    # FIRST print before any other code
    import sys
    import os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)  # Line buffering
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)

    model_runner = ModelRunner(config, rank, event)
    model_runner.loop()


class LLMEngine:
    def __init__(self, config: dict):
        self.config = config
        world_size = config.get("world_size", 1)
        ctx = mp.get_context("spawn")
        self.processes = []
        self.events = []
        for i in range(1, world_size):
            event = ctx.Event()
            process = ctx.Process(target=worker_process, args=(config, i, event))
            self.events.append(event)
            self.processes.append(process)
            process.start()
        # start the engine only on the master thread with rank = 0
        self.model_runner = ModelRunner(config, rank=0, event=self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.get("model_name_or_path", "gpt2"))
        
        # scheduler needs to init after model_runner: when world_size > 1,
        # ModelRunner.__init__ calls dist.init_process_group() which is a
        # collective barrier — rank-0 blocks until all worker ranks have joined.
        # The scheduler should only be created after that rendezvous completes.
        # When world_size == 1 there is no barrier and no real dependency.
        self.scheduler = Scheduler(
            max_num_sequences=config.get("max_num_sequences", 16),
            max_num_batched_tokens=config.get("max_num_batched_tokens", 1024),
            max_cached_blocks=config.get("max_cached_blocks", 1024),
            block_size=config.get("block_size", 256),
            eos=config.get("eos", 50256)
        )
        # prefix_scope_v1: Qwen3 single-GPU path implemented here.
        self.scheduler.block_manager.enable_prefix_caching = (
            config.get('enable_prefix_caching', True)
            and config.get('world_size', 1) == 1
            and self.model_runner.model.__class__.__name__ == 'Qwen3ForCausalLM'
        )
        # chunk_scheduler_install_v1
        if config.get("enable_chunked_prefill", False):
            assert config.get("world_size", 1) == 1, "Chunk path currently supports one GPU"
            assert self.model_runner.model.__class__.__name__ == "Qwen3ForCausalLM"
            from myvllm.engine.chunk_scheduler import ChunkScheduler
            self.scheduler = ChunkScheduler(
                self.scheduler, chunk_size=config.get("prefill_chunk_size", 256),
                decode_first=config.get("chunk_decode_first", True))

        atexit.register(self.exit)


    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for process in self.processes:
            process.join()

    # call scheduler to schedule the next batch
    # return scheduled sequences and whether it is for prefilling
    # call model_runner.run() to run the model
    # call postprocessor to process the outputs and update sequences and update block manager
    # 完成一次调度、执行和提交
    def step(self) -> tuple[list[tuple[int, list[int]]], int, bool]:
        # chunk_step_dispatch_v1
        # 开启 Chunk 时切换到混合调度路径
        if self.config.get('enable_chunked_prefill', False):
            # 导入 chunked_step 所需接口
            from myvllm.engine.chunk_runtime import chunked_step
            # 返回本次计算结果
            return chunked_step(self)
        # 由调度器决定本轮谁计算
        scheduled_sequences, is_prefill = self.scheduler.schedule()
        # 设置本轮处理量
        num_processed_tokens = 0
        # 本步选中的请求未就绪或为空时执行此分支
        if not scheduled_sequences:
            # 返回[]、本轮处理量、是否包含 Prefill
            return [], num_processed_tokens, is_prefill
        # run the model
        # 将本轮请求交给模型执行器
        outputs = self.model_runner.call("run", scheduled_sequences, is_prefill)
        # Move outputs to CPU and convert them to a list
        # 模型或请求输出已建立时执行此分支
        if outputs is not None:
            # 设置模型或请求输出
            outputs = outputs.cpu().tolist()
        # postprocess the outputs
        # publish_computed_prefix_v1
        # 逐个处理computed_seq
        for computed_seq in scheduled_sequences:
            # 前向结束后才能发布完整前缀页
            self.scheduler.block_manager.mark_computed(computed_seq)
        # 提交真实输出并回收已完成请求的页引用
        self.scheduler.postprocess(scheduled_sequences, outputs)

        # 设置模型或请求输出
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in scheduled_sequences if seq.is_finished]
        # 设置本轮处理量
        num_processed_tokens = sum(len(seq) for seq in scheduled_sequences) if is_prefill else len(scheduled_sequences)

        # 返回模型或请求输出、本轮处理量、是否包含 Prefill
        return outputs, num_processed_tokens, is_prefill


    # add prompt string to the waiting queue by first transforming it to Sequence object
    # 分词后创建普通引擎请求
    def add_prompt(self, prompt: str, sampling_params: SamplingParams) -> None:
        # 把请求放入等待队列
        self.scheduler.add_sequence(Sequence(token_ids=self.tokenizer.encode(prompt), block_size=self.config['block_size'],sampling_params=sampling_params))

    # given a list of prompts
    # add_prompt for each prompt
    # call step until all sequences are finished
    # return the generated texts
    # 运行请求直到达到固定输出配额
    def generate(self, prompts: list[str], sampling_params: SamplingParams) -> list[str]:
        # 逐个处理prompt
        for prompt in prompts:
            # 分词后创建普通引擎请求
            self.add_prompt(prompt, sampling_params)
        # 初始化已完成请求的输出
        generated_tokens = {}
        # 反复调度直到等待与运行请求全部结束
        while not self.scheduler.is_finished():
            # 设置本步开始时间
            start_t = time.time()
            # 设置模型或请求输出、本轮处理量、是否包含 Prefill
            outputs, num_processed_tokens, is_prefill = self.step()
            # 设置本步结束时间
            end_t = time.time()
            # 设置本步耗时
            running_time = end_t - start_t + 1e-10
            # 按 Prefill 或 Decode 选择处理方式
            if is_prefill:
                # 输出本轮进度或诊断信息
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during prefilling")
            # 处理上述条件不成立的情况
            else:
                # 输出本轮进度或诊断信息
                print(num_processed_tokens, 'number of processed tokens', num_processed_tokens/running_time, "tokens/sec during decoding")
            # 记录本轮刚完成的请求输出
            generated_tokens.update({seq_id: tokens for seq_id, tokens in outputs})

        # 设置已完成请求的输出
        generated_tokens = [generated_tokens[seq_id] for seq_id in sorted(generated_tokens.keys())]
        # 设置本轮返回值
        output = {'text': [self.tokenizer.decode(tokens) for tokens in generated_tokens], 'token_ids': generated_tokens}
        # 返回本轮返回值
        return output
