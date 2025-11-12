import asyncio
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Any, List, Optional

import httpx
from cc_agent import load_dataset
from rich.console import Console

from agentlightning import configure_logger
from agentlightning.adapter import LlmProxyTraceToTriplet
from agentlightning.llm_proxy import LLMProxy, ModelConfig
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.types import Dataset, Rollout

console = Console()


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


@contextmanager
def vllm_server(
    model_path: str,
    port: int,
    startup_timeout: float = 300.0,
    terminate_timeout: float = 10.0,
    max_model_len: int = 32768,
    gpu_memory_utilization: float = 0.7,
    quantization: Optional[str] = "bitsandbytes",
    auto_tool_choice: bool = True,
    tool_call_parser: Optional[str] = "hermes",
):
    """Serves a vLLM model from command line.

    Args:
        model_path: The path to the vLLM model. It can be either a local path or a Hugging Face model ID.
        port: The port to serve the model on.
        startup_timeout: The timeout for the server to start.
        terminate_timeout: The timeout for the server to terminate.
        max_model_len: The maximum model length.
        gpu_memory_utilization: The GPU memory utilization for the server. Set it lower to avoid OOM.
        quantization: The quantization method.
        auto_tool_choice: Whether to enable auto tool choice.
        tool_call_parser: The tool call parser to use.
    """
    proc: Optional[subprocess.Popen[bytes]] = None
    try:
        vllm_serve_args = [
            "--max-model-len",
            str(max_model_len),
            "--port",
            str(port),
        ]
        if quantization is not None:
            vllm_serve_args.append("--quantization")
            vllm_serve_args.append(quantization)
        if auto_tool_choice:
            vllm_serve_args.append("--enable-auto-tool-choice")
        if tool_call_parser is not None:
            vllm_serve_args.append("--tool-call-parser")
            vllm_serve_args.append(tool_call_parser)

        proc = subprocess.Popen(["vllm", "serve", model_path, *vllm_serve_args])

        # Wait for the server to be ready
        url = f"http://localhost:{port}/health"
        start = time.time()
        client = httpx.Client()

        while True:
            try:
                if client.get(url).status_code == 200:
                    break
            except Exception:
                result = proc.poll()
                if result is not None and result != 0:
                    raise RuntimeError("Server exited unexpectedly.") from None
                time.sleep(0.5)
                if time.time() - start > startup_timeout:
                    raise RuntimeError(f"Server failed to start in {startup_timeout} seconds.") from None

        yield f"http://localhost:{port}/v1"
    finally:
        # Terminate the server
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(terminate_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()


async def run_epoch(
    epoch: int,
    store: LightningStore,
    model_path: str,
    train_dataset: Any,
    llm_proxy: LLMProxy,
    data_adapter: LlmProxyTraceToTriplet,
    triplet_fraction: float,
) -> str:
    console.print(f"\n[bold red][Algo][/bold red] Starting epoch {epoch}")

    with vllm_server(
        model_path, _find_available_port(), quantization=None, tool_call_parser="qwen3_coder", max_model_len=128 * 1024
    ) as server_address:
        llm_proxy.update_model_list(
            [
                ModelConfig(
                    model_name="claude-sonnet-4-5-20250929",
                    litellm_params={
                        "model": f"hosted_vllm/{model_path}",
                        "api_base": server_address,
                    },
                ),
                ModelConfig(
                    model_name="claude-haiku-4-5-20251001",
                    litellm_params={
                        "model": f"hosted_vllm/{model_path}",
                        "api_base": server_address,
                    },
                ),
            ]
        )

        llm_proxy.restart()

        resources_update = await store.add_resources({"llm": llm_proxy.as_resource(model="local")})

        rollouts: List[Rollout] = []
        for _ in range(2):
            rollouts.append(
                await store.enqueue_rollout(
                    input=train_dataset[0], mode="train", resources_id=resources_update.resources_id
                )
            )

        console.print(f"[bold red][Algo][/bold red] Enqueued {len(rollouts)} rollouts")

        # Wait for the tasks to complete
        completed_rollouts: List[Rollout] = []

        while True:
            completed_rollouts = await store.wait_for_rollouts(
                rollout_ids=[rollout.rollout_id for rollout in rollouts],
                timeout=0.0,  # Timeout must be a very small value to avoid blocking the store server
            )
            if len(completed_rollouts) >= len(rollouts):
                console.print(f"[bold red][Algo][/bold red] Received all {len(rollouts)} rollouts")
                break
            console.print(
                f"[bold red][Algo][/bold red] Received {len(completed_rollouts)} rollouts, waiting for more..."
            )
            await asyncio.sleep(5.0)

    for rollout in completed_rollouts:
        spans = await store.query_spans(rollout.rollout_id, "latest")
        triplets = data_adapter.adapt(spans)

        # Logging the prompt and response lengths and rewards for debugging
        prompt_lengths = [len(t.prompt["token_ids"]) if t.prompt["token_ids"] else 0 for t in triplets]
        response_lengths = [len(t.response["token_ids"]) if t.response["token_ids"] else 0 for t in triplets]
        console.print(
            f"[bold red][Algo][/bold red] Rollout {rollout.rollout_id} has {len(spans)} spans, yielding {len(triplets)} triplets. "
            f"Prompt lengths: {prompt_lengths}. Response lengths: {response_lengths}. "
            f"Rewards are: {[t.reward for t in triplets]}"
        )

    return f"{model_path}_{epoch+1}"


async def run_algorithm(store: LightningStore, model_path: str, num_epochs: int, train_triplet_fraction: float) -> None:
    """An example SFT algorithm that communicates with rollout runners via the store.

    Args:
        store: The LightningStoreClient instance.
    """

    llm_proxy = LLMProxy(
        port=_find_available_port(),
        litellm_config={
            "general_settings": {
                "master_key": os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
            }
        },
        store=store,
    )
    data_adapter = LlmProxyTraceToTriplet()
    for epoch in range(num_epochs):
        train_dataset = load_dataset(epoch=epoch)
        model_path = await run_epoch(
            epoch=epoch,
            store=store,
            model_path=model_path,
            train_dataset=train_dataset,
            llm_proxy=llm_proxy,
            data_adapter=data_adapter,
            triplet_fraction=train_triplet_fraction,
        )

    console.print(f"[bold red][Algo][/bold red] Final model path: {model_path}")


if __name__ == "__main__":
    configure_logger()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--store_address", type=str, default="http://localhost:4747", help="The address of the LightningStore server."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        help="The model name or path for the LLM backend.",
    )
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument(
        "--train_triplet_fraction",
        type=float,
        default=0.8,
    )
    args = parser.parse_args()

    store = LightningStoreClient(args.store_address)

    asyncio.run(
        run_algorithm(
            store=store,
            model_path=args.model_name_or_path,
            num_epochs=args.num_epochs,
            train_triplet_fraction=args.train_triplet_fraction,
        )
    )
