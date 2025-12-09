from __future__ import annotations
from difflib import SequenceMatcher
from typing import TypedDict
import unidiff
import base64

from utils.docker_runtime import CommandResult, Runtime
from utils.type import ClaudeCodeStep, ClaudeCodeTraj, TrajSlice

class TajectoryProcessor:
    mapping = {
        "Read": "Localization",
        "Glob": "Localization",
        "Grep": "Localization",
        "Write": "Reproduction",
        "Edit": "Edit",
        "BashAfterEdit": "Validation",
        "Summary|Result": "Result",
    }

    @staticmethod
    def split(traj: ClaudeCodeTraj, slice_each_edit: bool = True) -> list[TrajSlice]:
        slices: list[TrajSlice] = []
        cur_act = "Localization"
        next_act = ""
        start = -1
        end = -1
        for idx in range(len(traj)):
            if traj[idx]['type'] == 'result':
                if traj[idx-1]['type'] == "assistant" and traj[idx-1]["message"]["content"][0]["type"] == "text":
                    slices.append({
                        "process": cur_act,
                        "step_range": (start, idx - 2),
                        "content": traj[start: idx - 1]
                    })
                    slices.append({
                        "process": "Result",
                        "step_range": (idx - 1, idx),
                        "content": traj[idx - 1: idx + 1]
                    })
                else:
                    slices.append({
                        "process": cur_act,
                        "step_range": (start, idx - 1),
                        "content": traj[start: idx]
                    })
                    slices.append({
                        "process": "Result",
                        "step_range": (idx, idx),
                        "content": traj[idx: idx + 1]
                    })
                break
            if traj[idx]['type'] == 'system':
                continue
            else:
                if start == -1:
                    start = idx
            if traj[idx]['type'] == 'user':
                continue
            if traj[idx]['type'] == 'assistant':
                if traj[idx]["parent_tool_use_id"] is not None:
                    # is subtask of sugagent
                    continue
                assert len(traj[idx]["message"]["content"]) == 1, traj[idx]["message"]["content"]
                if traj[idx]["message"]["content"][0]["type"] == "text":
                    continue
                content = traj[idx]["message"]["content"][0]
                if content["type"] == "tool_use":
                    # List events
                    if cur_act == "Edit" and content["name"] == "Bash":
                        next_act = "Validation"
                    elif cur_act == "Validation" and content["name"] == "Bash":
                        continue
                    elif content["name"] == "Write" and cur_act != "Reproduction":
                        next_act = "Reproduction"
                    elif content["name"] in {"Read", "Write", "Glob"} and cur_act != "Localization":
                        next_act = "Localization"
                    elif content["name"] == "Edit" and cur_act != "Edit":
                        next_act = "Edit"
                    elif slice_each_edit and content["name"] == "Edit" and cur_act == "Edit":
                        # We take each edit action as one slice in this setting
                        next_act = "Edit"
                    else:
                        # If no event to trigger, do not update slices list
                        continue
                    
                    for end in range(idx, 0, -1):
                        if traj[end]['type'] != 'assistant':
                            break
                    
                    if len(slices) >= 2 and cur_act == "Reproduction"\
                        and slices[-1]["process"] == "Localization"\
                        and slices[-2]["process"] == "Reproduction":
                        # merge
                        slices[-2]["step_range"] = (slices[-2]["step_range"][0], end)
                        slices[-2]["content"] = traj[slices[-2]["step_range"][0]: end + 1]
                        slices.pop() # default last
                    
                    else:
                        slices.append({
                            "process": cur_act,
                            "step_range": (start, end),
                            "content": traj[start: end + 1]
                        })
                    start = end + 1
                    cur_act = next_act
        return slices

    @staticmethod
    def extract_err(traj: ClaudeCodeTraj) -> list[TrajSlice]:
        err_list: list[TrajSlice] = []
        for idx in range(len(traj)):
            if traj[idx]['type'] == 'user' and "<tool_use_error>" in traj[idx]["message"]["content"][0]["content"]:
                assert len(traj[idx]["message"]["content"]) == 1, traj[idx]["message"]["content"]
                begin = None
                for begin in range(idx - 1, 0, -1):
                    if traj[begin]['type'] == 'user':
                        break
                if begin is not None:
                    err_list.append({
                        "process": "InvalidToolCall", # In [ToolFormatError, ToolNameError, StringReplaceNoChange, StringReplaceNotFound]
                        "step_range": (begin + 1, idx),
                        "content": traj[begin + 1: idx + 1]
                    })
        return err_list

class RewardEstimatorWholeSlice:
    '''
    This RewardEstimator assigns all the steps in a whole Slice the same reward.
    '''

    reward_range = {
        "Localization": "{ -1 } | (0, 1]",
        "Reproduction": "{ -1, 1 }",
        "Edit": "[-1, 1]",
        "Validation": "{ 1 }",
        "Result": "{ -1, 1 }",
        "InvalidToolCall": "{ -1 }"
    }

    class ParsedPatch(TypedDict):
        start: int
        end: int
        deleted_lines: list[str]
        added_lines: list[str]

    class RewardReturnType(TypedDict):
        traj: TrajSlice
        keysteps: ClaudeCodeTraj
        reward: float # [-1.0 , 1.0] in practice

    def __init__(self, 
                 container: Runtime, 
                 reproduction_file: str,
                 solution_patch: str,
                 gold_patch: str):
        
        self.container: Runtime = container
        self.reproduction_file: str = reproduction_file
        self.solution_patch: str = solution_patch
        self.gold_patch: str = gold_patch
        self.last_sim: float = 0.0
        self.last_diff: str = ""
        self.parsed_gold_patch: dict[str, list[RewardEstimatorWholeSlice.ParsedPatch]] = self._parse_patch(self.gold_patch)
        self.total_lines_gold_patch: int = 0
        self.sorted_added_lines: str = ""
        for file_path in sorted(self.parsed_gold_patch.keys()):
            for loc in sorted(self.parsed_gold_patch[file_path], key = lambda x: x["start"]):
                self.total_lines_gold_patch += loc["end"] - loc["start"]
                self.sorted_added_lines += "\n".join(loc["added_lines"]) + "\n"
        self.container.send_command("git stash; git reset --hard HEAD;")
        self.container.send_command("""perl -v &> /dev/null || (if command -v apt-get &> /dev/null; then apt-get update && apt-get install -y perl; elif command -v yum &> /dev/null; then yum install -y perl; elif command -v dnf &> /dev/null; then dnf install -y perl; elif command -v apk &> /dev/null; then apk add perl; elif command -v pacman &> /dev/null; then pacman -S --noconfirm perl; else echo "No supported package manager found" && exit 1; fi)""")

    @property
    def current_patch(self) -> str:
        return (self.container
                .send_command("git --no-pager diff HEAD --diff-filter=M --text")
                .output
                .replace("git --no-pager diff HEAD --diff-filter=M --text\n", ""))

    def read_file(self, file_path: str) -> str:
        return (self.container
                .send_command(f"cat {file_path}")
                .output
                .replace(f"cat {file_path}\n", ""))

    def write_file(self, file_path: str, content: str) -> bool:
        return (self.container
                .send_command(f"cat > {file_path} <<'REPRODUCTION_FILE'\n{content}\nREPRODUCTION_FILE\n")
                .metadata.exit_code == 0)

    def apply_patch(self, patch: str) -> bool:
        return (self.
                container.send_command(f"""git apply - <<'NEW_PATCH'\n{patch}\nNEW_PATCH""")
                .metadata.exit_code == 0)

    def string_replace(self, file_path: str, old_string: str, new_string: str) -> bool:
        # The old_string / new_string could be multi lines with special characters.
        # Use Perl for robust string replacement with special character handling
        
        # Escape file path for shell single quotes (bash/sh safe)
        # In single quotes, only ' needs escaping as '\''
        file_escaped = file_path.replace("'", "'\\''")
        
        # Base64 encode the strings to safely pass them through shell and Perl
        # This handles ALL special characters including quotes, newlines, brackets, etc.

        old_b64 = base64.b64encode(old_string.encode('utf-8')).decode('ascii')
        new_b64 = base64.b64encode(new_string.encode('utf-8')).decode('ascii')
        
        # Use Perl with -i for in-place editing and -0777 to slurp entire file
        # Decode base64 strings in Perl, then use \Q...\E to quote regex metacharacters
        # The replacement uses quotemeta on the decoded new string to escape special chars
        cmd = (
            f"perl -i -0777 -MMIME::Base64 -pe '"
            f"BEGIN{{"
            f"  use MIME::Base64;"
            f"  $old = decode_base64(q/{old_b64}/);"
            f"  $new = decode_base64(q/{new_b64}/);"
            f"}} "
            f"s/\\Q$old\\E/$new/s' '{file_escaped}'"
        )
        
        result: CommandResult = self.container.send_command(cmd)
        status = int(result.metadata.exit_code) == 0
        if not status:
            print(result.output)
        return status
    
    @staticmethod
    def _parse_patch(patch: str) -> dict[str, list[ParsedPatch]]:
        # use unidiff to get file_path : [(start1, end1, deleted_lines1, added_lines1), (start2, end2, deleted_lines2, added_lines2) ...] mapping
        # The added and deleted lines should not have + - and \n

        result: dict[str, list[RewardEstimatorWholeSlice.ParsedPatch]] = {}
        
        try:
            patch_set = unidiff.PatchSet(patch)
        except Exception:
            # If patch parsing fails, return empty dict
            return result
        
        for patched_file in patch_set:
            file_path = patched_file.path
            patches: list[RewardEstimatorWholeSlice.ParsedPatch] = []
            
            for hunk in patched_file:
                deleted_lines: list[str] = []
                added_lines: list[str] = []
                start = hunk.target_start
                end = hunk.target_start + hunk.target_length - 1
                
                for line in hunk:
                    if line.is_removed:
                        # Remove the leading '-' and trailing newline
                        deleted_lines.append(line.value.rstrip('\n'))
                    elif line.is_added:
                        # Remove the leading '+' and trailing newline
                        added_lines.append(line.value.rstrip('\n'))
                
                patches.append({
                    'start': start,
                    'end': end,
                    'deleted_lines': deleted_lines,
                    'added_lines': added_lines
                })
            
            if patches:
                result[file_path] = patches
        
        return result

    @staticmethod
    def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not intervals:
            return []
        res: list[tuple[int, int]] = []
        intervals = sorted(intervals)
        cur = list(intervals[0])
        for interval in intervals[1:]:
            if interval[0] <= cur[1]:
                if interval[1] > cur[1]:
                    cur[1] = interval[1]
            elif interval[0] - 1 == cur[1]:
                cur[1] = interval[1]
            else:
                res.append(tuple(cur))
                cur = list(interval)
        res.append(tuple(cur))
        return res

    def localization(self, slice: TrajSlice) -> RewardEstimatorWholeSlice.RewardReturnType:
        '''
        Reward: Recall of ground truth modified lines
        '''
        effective_lines: int = 0
        keysteps: ClaudeCodeTraj = []
        viewd_locs: dict[str, list[tuple[int, int]]] = {}
        for step_id in range(len(slice["content"])):
            step = slice["content"][step_id]
            if step["message"]["content"][0]["type"] == "tool_use" \
                and step["message"]["content"][0]["name"].lower() == "read"\
                and step["message"]["content"][0]["input"].get("file_path", None) is not None:
                    args: dict[str, str|int] = step["message"]["content"][0]["input"]
                    file_path: str = args["file_path"].replace("/testbed/", "")
                    if file_path not in self.parsed_gold_patch.keys():
                        continue
                    target_locs: list[RewardEstimatorWholeSlice.ParsedPatch] = self.parsed_gold_patch[file_path]
                    if args.get("offset", None) is not None and args.get("limit", None) is not None:
                        interval: tuple[int, int] = (args["offset"], args["offset"] + args["limit"])
                        if file_path in viewd_locs.keys():
                            viewd_locs[file_path].append(interval)
                        else:
                            viewd_locs[file_path] = [interval]
                        for target_loc in target_locs:
                            if target_loc["start"]  <= interval[1] <= target_loc["end"] or \
                                target_loc["start"]  <= interval[0] <= target_loc["end"]:
                                if slice["content"][step_id-1]["type"] == "assistant":
                                    keysteps.append(slice["content"][step_id-1])
                                keysteps.append(step)
                                break
                    else:
                        if slice["content"][step_id-1]["type"] == "assistant":
                            keysteps.append(slice["content"][step_id-1])
                        keysteps.append(step)
                        interval = (0, len(self.read_file(file_path).splitlines())-1)
                        viewd_locs[file_path] = [interval]
        for file_path in viewd_locs.keys():
            viewd_locs[file_path] = self._merge_intervals(viewd_locs[file_path])
            for target_loc in self.parsed_gold_patch[file_path]:
                for pred_loc in viewd_locs[file_path]:
                    if target_loc["start"]  <= pred_loc[1] <= target_loc["end"] or \
                        target_loc["start"]  <= pred_loc[0] <= target_loc["end"]:
                        effective_interval = (
                            max(target_loc["start"], pred_loc[0]),
                            min(target_loc["end"], pred_loc[1])
                        )
                        effective_lines += max(0, effective_interval[1] - effective_interval[0])

        return {
            "keysteps": keysteps,
            "reward": -1.0 if effective_lines == 0 else min(1.0, effective_lines/self.total_lines_gold_patch),
            "traj": slice,
        }
    
    def reproduction(self, slices: list[TrajSlice]) -> list[RewardEstimatorWholeSlice.RewardReturnType]:
        '''
        reward: whether reproduction.py exits with code from non zero to zero when gold_patch applied
        '''
        keysteps: ClaudeCodeTraj = []
        merged_slice: ClaudeCodeTraj = []
        for slice in slices:
            merged_slice.extend(slice["content"])
        for step_id in range(len(merged_slice)-1, 0, -1):
            if merged_slice[step_id]["message"]["content"][0]["type"] == "tool_use" and merged_slice[step_id]["message"]["content"][0]["name"].lower() == "write":
                if merged_slice[step_id-1]["type"] == "assistant":
                    keysteps.append(merged_slice[step_id-1])
                keysteps.append(merged_slice[step_id])
        
        command = "python /testbed/reproduction.py"
        self.write_file("/testbed/reproduction.py", self.reproduction_file)
        pre_patch_res: CommandResult = self.container.send_command(command)
        if int(pre_patch_res.metadata.exit_code) == 0:
            return [{
                "keysteps": keysteps,
                "reward": -1.0,
                "traj": slice,
            } for slice in slices]
        self.apply_patch(self.gold_patch)
        post_patch_res: CommandResult = self.container.send_command(command)
        if int(post_patch_res.metadata.exit_code) != 0:
            return [{
                "keysteps": keysteps,
                "reward": -1.0,
                "traj": slice,
            } for slice in slices]
        return [{
            "keysteps": keysteps,
            "reward": 1.0,
            "traj": slice,
        } for slice in slices]

    def _get_changed_lines(self, old_patch_parsed: dict[str, list[ParsedPatch]], new_patch_parsed: dict[str, list[ParsedPatch]]) -> dict[str, list[tuple[int, int]]]:
        res: dict[str, list[tuple[int, int]]] = {}
        for new_path in new_patch_parsed.keys():
            if new_path not in old_patch_parsed.keys():
                res[new_path] = [(loc["start"], loc["end"]) for loc in new_patch_parsed[new_path]]
            else:
                res[new_path] = []
                old_loc_index: dict[tuple[int, int], str] = {}
                for old_loc in old_patch_parsed[new_path]:
                    old_loc_index[(old_loc["start"], old_loc["end"])] = ("\n".join(old_loc["added_lines"])).strip()
                for new_loc in new_patch_parsed[new_path]:
                    cur_interval = (new_loc["start"], new_loc["end"])
                    if cur_interval not in old_loc_index.keys():
                        res[new_path].append(cur_interval)
                    else:
                        if ("\n".join(new_loc["added_lines"])).strip() != old_loc_index[cur_interval]:
                            res[new_path].append(cur_interval)
        return res
    
    @staticmethod
    def code_similarity(a: str, b: str) -> float:
        '''Ratcliff-Obershelp similarity algorithm'''
        return SequenceMatcher(None, a, b).ratio()
    
    def edit(self, slice: TrajSlice) -> RewardEstimatorWholeSlice.RewardReturnType:
        '''
        reward = old similarity - new similarity of added lines compared to ground truth added lines
        '''
        old_added_lines: str = ""
        old_modification: str = self.current_patch
        old_parsed_modification = self._parse_patch(old_modification)
        for file_patch in sorted(old_parsed_modification.keys()):
            for loc in sorted(old_parsed_modification[file_patch], key = lambda x: x["start"]):
                old_added_lines += "\n".join(loc["added_lines"]) + "\n"
        keysteps: ClaudeCodeTraj = []
        for step_id in range(len(slice["content"])):
            step = slice["content"][step_id]
            if step["message"]["content"][0]["type"] == "tool_use" \
                and step["message"]["content"][0]["name"].lower() == "edit" \
                and (not slice["content"][step_id+1]["message"]["content"][0].get("is_error", False)):
                if slice["content"][step_id-1]["type"] == "assistant":
                    keysteps.append(slice["content"][step_id-1])
                keysteps.append(step)
                self.string_replace(step["message"]["content"][0]["input"]["file_path"], 
                                    step["message"]["content"][0]["input"]["old_string"], 
                                    step["message"]["content"][0]["input"]["new_string"])
        new_modification: str = self.current_patch
        new_added_lines: str = ""
        new_parsed_modification = self._parse_patch(new_modification)
        modified_locs: dict[str, list[tuple[int,int]]] = self._get_changed_lines(old_parsed_modification, new_parsed_modification)
        # If no overlap, return -1.0
        has_overlap_with_gt = False
        for file_path in modified_locs.keys():
            for pred_locs in modified_locs[file_path]:
                if file_path not in self.parsed_gold_patch.keys():
                    continue
                else:
                    for gt_locs in self.parsed_gold_patch[file_path]:
                        if gt_locs["start"] <= pred_locs[0] <= gt_locs["end"] \
                            or gt_locs["start"] <= pred_locs[1] <= gt_locs["end"]:
                            has_overlap_with_gt = True
                            break
                if has_overlap_with_gt:
                    break
            if has_overlap_with_gt:
                break
        if not has_overlap_with_gt:
            return {
                "keysteps": keysteps,
                "reward": -1.0,
                "traj": slice
            }
        for file_patch in sorted(new_parsed_modification.keys()):
            for loc in sorted(new_parsed_modification[file_patch], key = lambda x: x["start"]):
                new_added_lines += "\n".join(loc["added_lines"]) + "\n"
        # Ratcliff-Obershelp similarity algorithm
        old_sim: float = self.code_similarity(self.sorted_added_lines, old_added_lines)
        new_sim: float = self.code_similarity(self.sorted_added_lines, new_added_lines)
        reward = new_sim - old_sim
        return {
            "keysteps": keysteps,
            "reward": reward,
            "traj": slice,
        }
    
    @staticmethod
    def validation(slices: list[TrajSlice]) -> list[RewardEstimatorWholeSlice.RewardReturnType]:
        return [{"keysteps": slice["content"],
             "reward": 1.0,
             "traj": slice} for slice in slices]
                
    @staticmethod
    def result(slice: TrajSlice) -> RewardEstimatorWholeSlice.RewardReturnType:
        reward: float = 1.0
        # These two cases are when tool call format is wrong 
        # so that tool call are parsed as text
        # which makes the agent early exit
        if "<tool_call>" in slice["content"][-1].get("result", ""):
            reward = -1.0
        elif len(slice["content"]) > 1 \
            and slice["content"][-2]["message"]["content"][0]["type"] == "text" \
            and "<tool_call>" in slice["content"][-2]["message"]["content"][0]["text"]:
            reward = -1.0
        return {
            "keysteps": slice["content"],
            "reward": reward,
            "traj": slice,
        }
    
    @staticmethod
    def assign_error_penalties(sequence: list[tuple[ClaudeCodeStep, float]]) -> list[tuple[ClaudeCodeStep, float]]:
        for step_id in range(len(sequence)):
            if sequence[step_id][0]["type"] == "user" and "<tool_use_error>" in sequence[step_id][0]["message"]["content"][0].get("content", ""):
                sequence[step_id] = (sequence[step_id][0], -1.0)
                sequence[step_id-1] = (sequence[step_id-1][0], -1.0)
                if sequence[step_id-2][0]["type"] == "assistant":
                    sequence[step_id-2] = (sequence[step_id-2][0], -1.0)
        return sequence
    
    @staticmethod
    def flattern(slices: list[RewardEstimatorWholeSlice.RewardReturnType]) -> list[tuple[ClaudeCodeStep, float]]:
        slices = sorted(slices, key = lambda x: x["traj"]["step_range"][0])
        res: list[tuple[ClaudeCodeStep, float]] = []
        for slice in slices:
            for step in slice["traj"]["content"]:
                res.append((step, slice["reward"]))
        return res

    def main(self, traj: ClaudeCodeTraj) -> list[tuple[ClaudeCodeStep, float]]:
        slices: list[TrajSlice] = TajectoryProcessor.split(traj)
        loc_slices: list[TrajSlice] = []
        repro_slices: list[TrajSlice] = []
        edit_slices: list[TrajSlice]= []
        validation_slices: list[TrajSlice]= []
        result_slice: TrajSlice | None = None
        for slice in slices:
            if slice["process"] == "Localization":
                loc_slices.append(slice)
            if slice["process"] == "Reproduction":
                repro_slices.append(slice)
            if slice["process"] == "Edit":
                edit_slices.append(slice)
            if slice["process"] == "Validation":
                validation_slices.append(slice)
            if slice["process"] == "Result":
                result_slice = slice
        assert result_slice is not None

        localization_reward_list: list[RewardEstimatorWholeSlice.RewardReturnType] = [self.localization(slice) for slice in loc_slices]
        reproduction_reward_list: list[RewardEstimatorWholeSlice.RewardReturnType] = self.reproduction(repro_slices)
        self.container.send_command("git stash; git reset --hard HEAD;")
        edit_reward_list: list[RewardEstimatorWholeSlice.RewardReturnType] = [self.edit(slice) for slice in edit_slices]
        validation_reward_list: list[RewardEstimatorWholeSlice.RewardReturnType] = self.validation(validation_slices)
        result_reward_list: list[RewardEstimatorWholeSlice.RewardReturnType] = [self.result(result_slice)]
        reward_sequence = self.flattern(localization_reward_list+reproduction_reward_list+edit_reward_list+validation_reward_list+result_reward_list)
        reward_sequence = self.assign_error_penalties(reward_sequence)
        return reward_sequence

