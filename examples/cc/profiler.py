import ast
import json
import math
import multiprocessing
import os
from datetime import datetime
from functools import partial
from typing import Dict, List

from tqdm import tqdm


def adapt_raw_gen_ai_request_span(span):
    """
    Extract the following fields from a raw_gen_ai_request span:
    - start_time: datetime.datetime
    - end_time: datetime.datetime
    - prompt: List[Dict]
    - response: Dict
    - prompt_token_ids: List[int]
    - response_token_ids: List[int]
    """
    entry = {
        "start_time": datetime.fromtimestamp(span["start_time"]),
        "end_time": datetime.fromtimestamp(span["end_time"]),
    }
    attrs = span["attributes"]
    prompt = ast.literal_eval(attrs["llm.hosted_vllm.messages"])
    choice = ast.literal_eval(attrs["llm.hosted_vllm.choices"])
    assert len(choice) == 1, f"Got {len(choice)} choices:\n{choice}"
    choice = choice[0]
    response = choice["message"]
    entry["prompt"] = prompt
    entry["response"] = response

    prompt_ids = ast.literal_eval(attrs["llm.hosted_vllm.prompt_token_ids"])
    response_ids = choice["token_ids"]
    entry["prompt_token_ids"] = prompt_ids
    entry["response_token_ids"] = response_ids
    return entry


class SpanAnalyzer:
    def __init__(self, spans: List[Dict], num_ticks: int = None):
        """
        Initialize with a list of time spans.
        Each span: {"start_time": datetime, "end_time": datetime}
        """
        self.spans = spans
        self.num_ticks = num_ticks
        self.map = None
        self.meta = []

    def analyse(self):
        maps = []
        global_start = min(span["start_time"] for span in self.spans)
        global_end = max(span["end_time"] for span in self.spans)
        total_duration = (global_end - global_start).total_seconds()
        num_ticks = math.ceil(total_duration) if self.num_ticks is None else self.num_ticks
        tick_duration = total_duration / num_ticks

        for span in self.spans:
            start_offset = (span["start_time"] - global_start).total_seconds()
            end_offset = (span["end_time"] - global_start).total_seconds()
            start_tick = int(start_offset / tick_duration)
            end_tick = int(end_offset / tick_duration)

            map_line = ["-"] * num_ticks
            for t in range(start_tick, min(end_tick + 1, num_ticks)):
                map_line[t] = "X"
            maps.append("".join(map_line))

            meta = {
                "duration": (span["end_time"] - span["start_time"]).total_seconds(),
                "sequence_id": span["sequence_id"],
            }
            if "reward" in span:
                meta["type"] = "reward"
                meta["reward"] = span["reward"]
            elif "response" in span:
                if span["response"]["tool_calls"]:
                    meta["type"] = f"tool[{span['response']['tool_calls'][0]['function']['name']}]"
                else:
                    meta["type"] = "text"
                meta["prompt_len"] = len(span["prompt_token_ids"])
                meta["response_len"] = len(span["response_token_ids"])
                meta["throughput"] = meta["response_len"] / meta["duration"]
                meta["start_proxy_cost"] = span["start_proxy_cost"]
                meta["end_proxy_cost"] = span["end_proxy_cost"]
                meta["proxy_cost (%)"] = span["proxy_cost (%)"]
            self.meta.append(meta)
        self.map = maps
        return tick_duration, total_duration

    def dump(self, save_path=None, verbose=False, tick_duration=1.0, total_duration=0.0):
        results = "Section: LLM Call Timeline in order\n"
        results += f"Tick Duration: {tick_duration:.2f} s | Total Duration: {total_duration:.2f} s\n"
        for maps, meta in zip(self.map, self.meta):
            results += f"{'S-' + str(meta['sequence_id']).zfill(3):>5}: [{maps}] | Type: {meta['type']:<15} | Duration: {meta['duration']:6.2f} s"
            if meta["type"] != "reward":
                results += f" | Prompt Len: {meta['prompt_len']:6} | Response Len: {meta['response_len']:6} | Throughput: {meta['throughput']:7.2f} tok/s"
            else:
                results += f" | Reward: {meta['reward']}"
            results += "\n"

        results += "\n\n\nSection: Proxy Cost Summary\n"
        for meta in self.meta:
            if meta["type"] == "reward":
                continue
            results += (
                f"{'S-' + str(meta['sequence_id']).zfill(3):>5}: Start Proxy Cost: {meta['start_proxy_cost']:<4.2f} s"
            )
            results += f" | End Proxy Cost: {meta['end_proxy_cost']:4.2f} s | Total Proxy Cost (%): {meta['proxy_cost (%)']:5.2f} %\n"

        if save_path:
            with open(save_path, "w") as f:
                f.write(results)
        if verbose:
            print(results)


def worker_func(instance_span_path, span_dir, save_dir, verbose, num_ticks):
    """Worker function to process a single span file."""
    try:
        spans, collected_entries = [], []
        save_path = None
        if save_dir:
            instance_id = instance_span_path.split("-")[:-2]
            save_path = os.path.join(save_dir, f"{'-'.join(instance_id)}.log")

        with open(os.path.join(span_dir, instance_span_path)) as f:
            for line in f:
                spans.append(json.loads(line))

        # Each request consists of 4 spans
        assert len(spans) % 4 == 1, f"Span count {len(spans)} is not in expected format."

        for offset in range(0, len(spans) - 1, 4):
            new_entry, start_time_span_map, end_time_span_map = {}, {}, {}
            sequence_id = spans[offset]["sequence_id"]

            for idx in range(4):
                span_idx = idx + offset
                span = spans[span_idx]

                assert sequence_id == span["sequence_id"]

                st = datetime.fromtimestamp(span["start_time"])
                et = datetime.fromtimestamp(span["end_time"])
                start_time_span_map[span["name"]] = st
                end_time_span_map[span["name"]] = et

                if span["name"] == "raw_gen_ai_request":
                    # Shared field:
                    # - sequence_id
                    new_entry["sequence_id"] = sequence_id
                    # for raw_gen_ai_request span, extract fields:
                    # - start_time
                    # - end_time
                    # - prompt
                    # - response
                    # - prompt_token_ids
                    # - response_token_ids
                    new_entry.update(adapt_raw_gen_ai_request_span(span))

            assert len(new_entry) > 0, f"Failed to adapt spans at offset {offset} in file {instance_span_path}"

            # Timing and Proxy Cost Logic
            start_time_span_map = dict(sorted(start_time_span_map.items(), key=lambda x: x[1]))
            end_time_span_map = dict(sorted(end_time_span_map.items(), key=lambda x: x[1]))

            # Sanity CHECK 2: start&end order of spans
            assert list(start_time_span_map.keys()) == [
                "litellm_request",
                "raw_gen_ai_request",
                "Received Proxy Server Request",
                "self",
            ], f"Unexpected start time span order: {list(start_time_span_map.keys())}"
            assert list(end_time_span_map.keys()) == [
                "self",
                "litellm_request",
                "raw_gen_ai_request",
                "Received Proxy Server Request",
            ], f"Unexpected end time span order: {list(end_time_span_map.keys())}"

            start_proxy_cost = (start_time_span_map["self"] - start_time_span_map["litellm_request"]).total_seconds()
            end_proxy_cost = (
                end_time_span_map["Received Proxy Server Request"] - end_time_span_map["self"]
            ).total_seconds()
            max_duration = max(
                [
                    (end_time_span_map[name] - start_time_span_map[name]).total_seconds()
                    for name in start_time_span_map.keys()
                ]
            )

            # Cannot pass this type of proxy cost check,
            # NOTE: the larger the generation time, the higher the end_proxy_cost
            # assert start_proxy_cost <= 0.1, f"Start proxy cost too high: {start_proxy_cost}, duration: {(end_time_span_map['Received Proxy Server Request'] - start_time_span_map['litellm_request']).total_seconds()}"
            # assert end_proxy_cost <= 0.5, f"End proxy cost too high: {end_proxy_cost}, duration: {(end_time_span_map['Received Proxy Server Request'] - start_time_span_map['litellm_request']).total_seconds()}"

            # assert start_proxy_cost / max_duration * 100 < 10, f"Start proxy cost too high: {start_proxy_cost}, max duration: {max_duration}, take {start_proxy_cost/max_duration*100:.2f}%"
            # assert end_proxy_cost / max_duration * 100 < 25, f"End proxy cost too high: {end_proxy_cost}, max duration: {max_duration}, take {end_proxy_cost/max_duration*100:.2f}%"
            new_entry["start_proxy_cost"] = start_proxy_cost
            new_entry["end_proxy_cost"] = end_proxy_cost
            new_entry["proxy_cost (%)"] = (start_proxy_cost + end_proxy_cost) / max_duration * 100
            collected_entries.append(new_entry)

        # collect reward span
        # Sanity CHECK 4: reward span
        span = spans[-1]
        assert span["name"] == "agentlightning.annotation", f"Last span is not reward span: {span}"
        assert "agentlightning.reward.0.value" in span["attributes"], f"Cannot find reward attribute in span: {span}"

        reward_value = span["attributes"]["agentlightning.reward.0.value"]
        collected_entries.append(
            {
                "start_time": datetime.fromtimestamp(span["start_time"]),
                "end_time": datetime.fromtimestamp(span["end_time"]),
                "reward": reward_value,
                "sequence_id": span["sequence_id"],
            }
        )

        # Analyze and Save
        analyzer = SpanAnalyzer(collected_entries, num_ticks=num_ticks)
        tick_duration, total_duration = analyzer.analyse()
        analyzer.dump(save_path=save_path, verbose=verbose, tick_duration=tick_duration, total_duration=total_duration)

        if save_path:
            with open(save_path, "a") as f:
                f.write("\n\n\nSection: Raw Entries (excluding reward entry)\n")

                filtered_entries = [
                    {
                        "sequence_id": entry["sequence_id"],
                        "start_time": entry["start_time"],
                        "end_time": entry["end_time"],
                        "prompt": entry["prompt"],
                        "response": entry["response"],
                    }
                    for entry in collected_entries[:-1]
                ]

                json.dump(filtered_entries, f, default=str, indent=2)

        return True
    except Exception as e:
        return f"Error processing {instance_span_path}: {e}"


def main(args):
    span_dir, save_dir = args.span_dir, args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    instance_span_paths = [f for f in os.listdir(span_dir) if f.endswith(".json")]

    # Use functools.partial to fix the constant arguments (dir paths, verbose flag)
    worker = partial(worker_func, span_dir=span_dir, save_dir=save_dir, verbose=args.verbose, num_ticks=args.num_ticks)

    # Use CPU count or a user-defined number of workers
    num_workers = min(len(instance_span_paths), multiprocessing.cpu_count())

    print(f"Profiling with {num_workers} workers...")

    with multiprocessing.Pool(processes=num_workers) as pool:
        # pool.imap allows us to wrap the execution in tqdm for a progress bar
        results = list(
            tqdm(pool.imap(worker, instance_span_paths), total=len(instance_span_paths), desc="Processing span files")
        )

    assert len(results) == len(instance_span_paths), "Some span files were not processed."
    # Check for errors in results
    errors = [r for r in results if isinstance(r, str)]
    if errors:
        print(f"\nCompleted with {len(errors)} errors:")
        for err in errors:
            print(err)


def legacy_main(args):
    span_dir, save_dir = args.span_dir, args.save_dir
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    instance_span_paths = [f for f in os.listdir(span_dir) if f.endswith(".json")]
    instance_ids = ["".join(f.split("-")[:-2]) for f in instance_span_paths]
    assert len(instance_ids) == len(set(instance_ids)), "Duplicate instance IDs found in span files."
    for instance_span_path in tqdm(instance_span_paths, desc="Processing span files"):
        result = worker_func(
            instance_span_path,
            span_dir=span_dir,
            save_dir=save_dir,
            verbose=args.verbose,
            num_ticks=args.num_ticks,
        )
        if isinstance(result, str):
            print(result)
        # Perf CHECK 5: agent execution time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--span_dir",
        type=str,
        required=True,
        help="Directory containing span JSON files.",
    )
    parser.add_argument("--save_dir", type=str, help="Directory to save profiling results.", default=None)
    parser.add_argument("--verbose", action="store_true", help="Whether to print profiling results.")
    parser.add_argument("--num_ticks", type=int, default=None, help="Number of ticks for timeline visualization.")
    parser.add_argument(
        "--use_legacy", action="store_true", help="Whether to use the legacy single-process main function."
    )
    args = parser.parse_args()
    if args.use_legacy:
        legacy_main(args)
    else:
        main(args)
