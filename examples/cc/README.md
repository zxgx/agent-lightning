# Training ANY LLM to Claude Code

This example wraps Claude Code as the agent to:
1. collect traces from agent execution;
2. train a hosted LLM with the traces ***🔨 Under development***

## Requirements
1. install [agentlightning](https://microsoft.github.io/agent-lightning/stable/tutorials/installation/)
2. `(uv) pip install swebench` for evaluation

## Trace collection
We support running Claude Code via two ways:
- Hosted LLM servers, supporting versatile customizations
- Official Claude Code

### From Hosted LLM server
1. Prepare an OpenAI-compatible server:
```bash
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --max-model-len 131072 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

2. Sanity check:
```bash
cd examples/cc

# Suppose the vllm server is running at localhost
python cc_agent \
    --model_name_or_path Qwen/Qwen3-Coder-30B-A3B-Instruct \
    --server_address http://localhost:8000/v1 \
    --dataset_path swe_debug.jsonl \
    --max_step 32 \
    --output_dir data_debug
```
We provide a small dataset `swe_debug.jsonl` which is a subset of [SWE-bench](https://huggingface.co/datasets/SWE-bench/SWE-bench) for sanity check.

The above commands will generate a `data_debug` dir, which contains two targets: (1) a Huggingface Dataset named `dataset-<instance_id>` and (2) a trace file named `stream_<instance_id>.jsonl`, where `instance_id` is a unique key of the SWE-bench samples.
The dataset showcases the versatile customization capability of agent-lightning. In particular, we support extracting prompt/response ids, logprobs from the vllm server.
The trace file is the conversation logs for claude code to tackle the SWE-bench instance.

In addition, there will be a `logs` dir, which is the output of the docker container executing agent calls.

### From official Claude Code
1. Prepare ANTHROPIC_API_KEY
```bash
export ANTHROPIC_API_KEY=sk-<your private key>
```

2. Sanity check
```bash
cd examples/cc
python cc_agent \
    --official \
    --dataset_path swe_debug.jsonl \
    --max_step 32 \
    --output_dir data_debug
```
As the underlying model is provided by Anthropic, we cannot obtain prompt/response ids and logprobs. However, we can still obtain a trace file named `<instance_id>.json` under `data_debug`.
