#!/bin/bash

# Env vars set by amulet:
# - WORLD_SIZE: node size
# - GPU_PER_NODE_COUNT: gpus per node
# - RANK & NODE_RANK: node index
# - LOCAL_RANK: None
# - MASTER_ADDR: node-0
# - MASTER_PORT: 9500

# On agent side:
# make sure GPU server port 4747 and 8765 are reachable, for example, through ssh tunnel.
# python rollout_runner.py --max_step 50 --n_runners 4

# On GPU side free port 4747 and 8765
# 1. launch store:
# agl store --port 4747

# 2. start trace collection
# bash collect_trace.sh \
# --model_tag  Qwen/Qwen3-4B-Instruct-2507 \
# --tool_call_parser  hermes \
# --num_samples  8 \
# --dataset_path  /mnt/input/datasets/swe100.jsonl

# example model tags and tool call parsers:
# Qwen/Qwen3-Coder-30B-A3B-Instruct - qwen3_coder
# Qwen/Qwen3-4B-Instruct-2507 - hermes

export NCCL_DEBUG=WARN
export HF_HOME=$HOME/.cache/huggingface


# Make it able to receive named cmd args, --model_tag, --tool_call_parser, --num_samples
model_tag=Qwen/Qwen3-4B-Instruct-2507
tool_call_parser=hermes
num_repeats=1
dataset_path=swe_debug.jsonl

output_dir=/mnt/input/agl_trace/debug
mkdir -p $output_dir

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_tag)
            model_tag="$2"
            shift 2
            ;;
        --tool_call_parser)
            tool_call_parser="$2"
            shift 2
            ;;
        --num_repeats)
            num_repeats="$2"
            shift 2
            ;;
        --dataset_path)
            dataset_path="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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
    --dataset_path $dataset_path \
    --num_repeats $num_repeats\
    --dataset_dump_path $output_dir/datasets \
    --span_dump_path $output_dir/spans 2>&1 | tee $output_dir/collect.log
