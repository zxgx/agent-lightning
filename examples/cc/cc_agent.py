import asyncio
import json
import os
import platform

if platform.system() == "Linux":
    import resource

from typing import Any, Dict, Literal, Optional

from evaluation import evaluate
from swebench.harness.utils import load_swebench_dataset
from utils.claude_code_controller import ClaudeController
from utils.logger import logger

from agentlightning import (
    InMemoryLightningStore,
    LitAgentRunner,
    LlmProxyTraceToTriplet,
    OtelTracer,
    Trainer,
    configure_logger,
)
from agentlightning.litagent import LitAgent
from agentlightning.llm_proxy import LLMProxy
from agentlightning.types import LLM, AttemptedRollout, NamedResources, ProxyLLM, Rollout, RolloutRawResult


def load_dataset(path: str = "swe_debug.jsonl", epoch: int = 0, limit: Optional[int] = None) -> Dict[str, Any]:
    instances = []
    with open(path) as f:
        for line in f:
            instance = json.loads(line)
            instance["epoch"] = epoch
            instances.append(instance)

    if limit is not None:
        instances = instances[:limit]
    return instances


class CodingAgent(LitAgent):
    def __init__(
        self,
        namespace: Literal["swebench", "starryzhang"] = "swebench",
        full_set: Literal["princeton-nlp/SWE-bench", "SWE-bench-Live/SWE-bench-Live"] = "princeton-nlp/SWE-bench",
        split: str = "test",
        max_step: int = 20,
        run_method: Literal["python", "cli"] = "cli",
        open_file_limit: int = 4096,
        cache_level: str = "env",  # ["none", "base", "env", "instance"]
        clean: bool = False,
        force_rebuild: bool = False,
        timeout: int = 1_800,  # in sec
        instance_image_tag: str = "latest",
        rewrite_reports: bool = False,
    ) -> None:
        super().__init__()
        self.namespace = namespace
        self.full_set = full_set
        self.split = split
        self.max_step = max_step
        self.run_method = run_method

        self.cache_level = cache_level
        self.clean = clean
        self.force_rebuild = force_rebuild
        self.timeout = timeout
        self.instance_image_tag = instance_image_tag
        self.rewrite_reports = rewrite_reports

        full_dataset = load_swebench_dataset(full_set, split)
        self.dataset = {each["instance_id"]: each for each in full_dataset}

        # run instances locally
        if platform.system() == "Linux":
            resource.setrlimit(resource.RLIMIT_NOFILE, (open_file_limit, open_file_limit))

    async def rollout_async(
        self, task: Dict[str, Any], resources: NamedResources, rollout: Rollout
    ) -> RolloutRawResult:
        run_id = f"epoch_{task.get('epoch', 0)}"
        image = f"{self.namespace}/sweb.eval.x86_64.{task['instance_id'].lower()}".replace("__", "_1776_")

        llm = resources.get("llm")
        assert llm is not None, "LLM resource is required for rollout."

        llm = self._strip_proxy_helper(llm, rollout)

        try:
            # 1. init container
            controller = ClaudeController(
                image, task, run_id, llm.endpoint, llm.api_key or os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy")
            )
            # 2. execute task
            prediction = controller.run_instance(task, max_step=self.max_step, run_method=self.run_method)
            del controller
        except Exception as e:
            logger(run_id, task["instance_id"], f"Exception during rollout: {e}")

        # 3. obtain rewards (evaluation result)
        reward = 0.0
        # empty patch
        if prediction["model_patch"] in ["", None]:
            return reward

        instance_id = prediction["instance_id"]

        result = evaluate(
            prediction,
            self.dataset[instance_id],
            self.cache_level,
            self.clean,
            self.force_rebuild,
            run_id,
            self.timeout,
            namespace=self.namespace,
            instance_image_tag=self.instance_image_tag,
            rewrite_reports=self.rewrite_reports,
        )

        # error patch
        if result is None:
            return reward

        report = result[0]
        # resolved/unresolved patch
        if report[instance_id]["resolved"]:
            reward = 1.0
        return reward

    def _strip_proxy_helper(self, proxy_llm: LLM, rollout: Rollout) -> LLM:
        """Convert [`ProxyLLM`][agentlightning.ProxyLLM] instances into concrete LLMs.

        It resolves ProxyLLM instances to their concrete LLM implementation
        by attaching the attempted rollout context. This is only used when the function
        signature accepts an `llm` parameter and strip_proxy is True.

        Args:
            proxy_llm: Candidate LLM resource.
            rollout: Rollout metadata that provides rollout and attempt identifiers.

        Returns:
            [`LLM`][agentlightning.LLM] with rollout context baked into the endpoint.

        Raises:
            ValueError: If the rollout is not an
                [`AttemptedRollout`][agentlightning.AttemptedRollout].
        """

        if not isinstance(proxy_llm, ProxyLLM):
            # Not a ProxyLLM, nothing to strip here.
            return proxy_llm

        # Rollout is still a Rollout here because API is not stabilized yet.
        # In practice, it must be an AttemptedRollout.
        if not isinstance(rollout, AttemptedRollout):
            raise ValueError("Rollout is not an AttemptedRollout.")

        return proxy_llm.with_attempted_rollout(rollout)


async def cc_agent_dry_run_sample(model_path, server_address) -> None:
    """Run a dry run of the cc agent on a single sample.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_dataset(limit=4)

    tracer = OtelTracer()
    runner = LitAgentRunner(tracer)
    store = InMemoryLightningStore()
    adapter = LlmProxyTraceToTriplet()

    llm_proxy = LLMProxy(
        port=12358,
        litellm_config={
            "general_settings": {
                "master_key": os.environ.get("ANTHROPIC_AUTH_TOKEN", "dummy"),
                # "forward_client_headers_to_llm_api": True,
            }
        },
    )
    llm_proxy.set_store(store)

    model_list = [
        {
            "model_name": "claude-sonnet-4-5-20250929",
            "litellm_params": {
                "model": f"hosted_vllm/{model_path}",
                "api_base": server_address,
            },
        },
        {
            "model_name": "claude-haiku-4-5-20251001",
            "litellm_params": {
                "model": f"hosted_vllm/{model_path}",
                "api_base": server_address,
            },
        },
    ]

    llm_proxy.update_model_list(model_list)
    # Restart the LLM proxy after backend model list update
    # If LLM proxy has never been started, it will be started
    llm_proxy.restart()

    # Put the LLM proxy address into the store as an address
    await store.add_resources(
        {
            "llm": llm_proxy.as_resource(model="local"),
        }
    )

    with runner.run_context(agent=CodingAgent(), store=store):
        rollout = await runner.step(
            dataset[0],
        )

        rollouts = await store.query_rollouts()
        spans = await store.query_spans(rollout.rollout_id)
        triplets = adapter.adapt(spans)


def cc_agent_dry_run_dataset():
    """Run a dry run of the cc agent on a small dataset.

    This is a simple test function that runs the math agent on the first 4 problems
    using a single worker. Useful for testing the setup and configuration.
    """
    dataset = load_dataset(limit=4)
    trainer = Trainer(
        n_runners=1,
        initial_resources={
            "llm": LLM(
                endpoint=os.environ["ANTHROPIC_BASE_URL"],
                api_key=os.environ["ANTHROPIC_AUTH_TOKEN"],
                model="local",
            )
        },
    )
    trainer.dev(CodingAgent(), dataset)


if __name__ == "__main__":
    configure_logger()
    asyncio.run(
        cc_agent_dry_run_sample(
            model_path="Qwen/Qwen3-Coder-30B-A3B-Instruct",
            server_address="http://GCRAZGDL1513.westus3.cloudapp.azure.com:8000/v1",
        )
    )
