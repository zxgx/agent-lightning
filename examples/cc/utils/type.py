from typing import Literal, TypedDict

CC_ALL_TOOLS = {
    "Bash",
    "Edit",
    "Glob",
    "Grep",
    "NotebookEdit",
    "NotebookRead",
    "Read",
    "SlashCommand",
    "Task",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
    "Write",
}


class DatasetConfig(TypedDict):
    dataset_dir: str
    namespace: Literal["swebench", "starryzhang"]
    full_set: Literal["princeton-nlp/SWE-bench", "SWE-bench-Live/SWE-bench-Live"]
    split: str


class CCConfig(TypedDict):
    tools: list[str]
    user_prompt: str


class RumtimeConfig(TypedDict):
    epochs: int
    max_step: int
    workers: int
    num_samples: int
    run_id: str
    run_method: Literal["cli", "python"]
    overwrite: bool


class AgentConfig(TypedDict):
    dataset: DatasetConfig
    agent: CCConfig
    runtime: RumtimeConfig


class SWEbenchInput(TypedDict):
    instance_id: str
    model_patch: str
    model_name_or_path: str


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ToolCallContent(TypedDict):
    type: Literal["tool_use"]
    name: str
    input: dict[str, str]


class ToolResultContent(TypedDict):
    type: Literal["tool_result"]
    content: str
    is_error: bool


class ClaudeCodeMessage(TypedDict):
    type: str
    content: list[TextContent | ToolCallContent | ToolResultContent]


class ClaudeCodeStep(TypedDict):
    type: str
    message: ClaudeCodeMessage

SliceType = Literal["Localization", "Reproduction", "Edit", "Validation", "Result"]

ClaudeCodeTraj = list[ClaudeCodeStep]


class AgentResult(SWEbenchInput):
    trajectory: ClaudeCodeTraj

class TrajSlice(TypedDict):
    process: SliceType
    step_range: tuple[int, int]
    content: ClaudeCodeTraj


class RewardReturnType(TypedDict):
    traj: TrajSlice
    keysteps: ClaudeCodeTraj
    reward: float # [-1.0 , 1.0] in practice
