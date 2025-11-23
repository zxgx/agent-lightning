import asyncio
import multiprocessing
from typing import Any, Dict

from cc_agent import CodingAgent
from rich.console import Console

from agentlightning import LitAgentRunner, configure_logger
from agentlightning.store import LightningStore, LightningStoreClient
from agentlightning.tracer import OtelTracer

console = Console()


def run_rollout(*, max_step: int, store: LightningStore, worker_id: int) -> None:
    tracer = OtelTracer()
    runner = LitAgentRunner[Dict[str, Any]](tracer)

    console.print(f"[bold green]Runners: [/bold green] Rollout runner {worker_id} started.")

    with runner.run_context(agent=CodingAgent(max_step=max_step), store=store, worker_id=worker_id):
        asyncio.run(runner.iter())


def spawn_runners(*, store: LightningStore, n_runners: int, max_step: int) -> None:
    runners = [
        multiprocessing.Process(
            target=run_rollout, kwargs={"max_step": max_step, "store": store, "worker_id": worker_id}
        )
        for worker_id in range(n_runners)
    ]
    for runner in runners:
        runner.start()

    for runner in runners:
        runner.join()


if __name__ == "__main__":
    configure_logger()
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--store_address", type=str, default="http://localhost:4747", help="The address of the LightningStore server."
    )
    parser.add_argument("--max_step", type=int, default=5, help="Maximum steps per instance.")
    parser.add_argument("--n_runners", type=int, default=4, help="Number of rollout runners to spawn.")

    args = parser.parse_args()
    store = LightningStoreClient(args.store_address)
    spawn_runners(store=store, n_runners=args.n_runners, max_step=args.max_step)
