from typing import Any, Dict, Optional, Union

from litellm.integrations.custom_logger import CustomLogger

from agentlightning.llm_proxy import _get_pre_call_data


class AddLogprobs(CustomLogger):
    """LiteLLM logger hook to request logprobs from vLLM.

    This mutates the outgoing request payload to include `logprobs=1`
    for backends that support logprobs return (e.g., vLLM).
    """

    async def async_pre_call_hook(self, *args: Any, **kwargs: Any) -> Optional[Union[Exception, str, Dict[str, Any]]]:
        """Async pre-call hook to adjust request payload.

        Args:
            args: Positional args from LiteLLM.
            kwargs: Keyword args from LiteLLM.

        Returns:
            Either an updated payload dict or an Exception to short-circuit.
        """
        try:
            data = _get_pre_call_data(args, kwargs)
        except Exception as e:
            return e

        # Ensure logprobs are requested from the backend when supported.
        return {**data, "logprobs": 1}
