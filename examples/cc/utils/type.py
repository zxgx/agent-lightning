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


class SWEbenchInput(TypedDict):
    instance_id: str
    model_patch: str
    model_name_or_path: str

class TextContent(TypedDict):
    type = "text"
    text: str

class ToolCallContent(TypedDict):
    type = "tool_use"
    name: str
    input: dict

class ToolResultContent(TypedDict):
    type = "tool_result"
    content: str
    is_error: bool

class ClaudeCodeMessage(TypedDict):
    type: str
    content: list [ TextContent | ToolCallContent | ToolResultContent ]

class ClaudeCodeStep(TypedDict):
    type: str
    message: ClaudeCodeMessage

# SliceType = Literal["Localization", "Reproduction", "Edit", "Validation", "Result"]

ClaudeCodeTraj = list[ClaudeCodeStep]

class AgentResult(SWEbenchInput):
    trajectory: ClaudeCodeTraj

# class TrajSlice(TypedDict):
#     process: SliceType
#     step_range: tuple[int, int]
#     content: ClaudeCodeTraj


# class RewardReturnType(TypedDict):
#     traj: TrajSlice
#     keysteps: ClaudeCodeTraj
#     reward: float # [-1.0 , 1.0] in practice