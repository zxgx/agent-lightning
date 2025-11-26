#!/bin/bash

# Env vars set by amulet:
# - WORLD_SIZE: node size
# - GPU_PER_NODE_COUNT: gpus per node
# - RANK & NODE_RANK: node index
# - LOCAL_RANK: None
# - MASTER_ADDR: node-0
# - MASTER_PORT: 9500

# on agent side:
# python rollout_runner.py --max_step 50 --n_runners 4

# 1. on GPU side
# agl store --port 4747

# 2. on GPU side
# - Qwen/Qwen3-Coder-30B-A3B-Instruct - qwen3_coder
# - Qwen/Qwen3-4B-Instruct-2507 - hermes

export NCCL_DEBUG=WARN
# export HF_HOME=/mnt/input/cache/huggingface

model_tag=Qwen/Qwen3-Coder-30B-A3B-Instruct
tool_call_parser=qwen3_coder
model_id=qwen3_coder_30b

secondary_dp_option=""
if [ $NODE_RANK -gt 0 ]; then
    secondary_dp_option="--headless --data-parallel-start-rank $NODE_RANK"
fi

parallel_config="
--tensor-parallel-size $GPU_PER_NODE_COUNT \
--data-parallel-size $WORLD_SIZE --data-parallel-size-local 1 \
--data-parallel-address $MASTER_ADDR --data-parallel-rpc-port 13345 \
$secondary_dp_option
"

python parallel_trace_collection.py \
    --access_host localhost \
    --model_name_or_path $model_tag \
    --vllm_serve_args_str "--enable-auto-tool-choice --tool-call-parser $tool_call_parser $parallel_config" \
    --dataset_path swe_100.jsonl \
    --dataset_dump_path /mnt/input/agl_trace/$model_id/datasets \
    --span_dump_path /mnt/input/agl_trace/$model_id/spans 2>&1 | tee /mnt/input/agl_trace/collect.log
