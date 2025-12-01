'''
This file is still under development...
Not used...
'''

import json
from typing import TypedDict
import unidiff
import base64

from utils.docker_runtime import Runtime
from utils.type import ClaudeCodeStep, ClaudeCodeTraj, RewardReturnType, TrajSlice

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
    def split(traj: ClaudeCodeTraj) -> list[TrajSlice]:
        slices = []
        cur_act = "Localization"
        next_act = ""
        start = -1
        end = -1
        has_reproduction = False
        for idx in range(len(traj)):
            if traj[idx]['type'] == 'result':
                slices.append({
                    "process": cur_act,
                    "step_range": [start, idx - 2],
                    "content": traj[start: idx - 1]
                })
                slices.append({
                    "process": "Result",
                    "step_range": [idx - 1, idx],
                    "content": [traj[idx - 1: idx + 1]]
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
                    if not has_reproduction:
                        if content["name"] != "Write":
                            continue
                        else:
                            has_reproduction = True
                    if cur_act == "Edit" and content["name"] == "Bash":
                        next_act = "Validation"
                    elif cur_act == "Validation" and content["name"] == "Bash":
                        continue
                    elif content["name"] == "Write":
                        if cur_act == "Reproduction":
                            continue
                        else:
                            next_act = "Reproduction"
                    elif content["name"] in {"Read", "Write", "Glob"}:
                        if cur_act == "Localization":
                            continue
                        else:
                            next_act = "Localization"
                    elif content["name"] == "Edit":
                        if cur_act == "Edit":
                            continue
                        else:
                            next_act = "Edit"
                    else:
                        continue
                    
                    for end in range(idx, 0, -1):
                        if traj[end]['type'] != 'assistant':
                            break
                    
                    if len(slices) > 2 and cur_act == "Reproduction"\
                        and slices[-1]["process"] == "Localization"\
                        and slices[-2]["process"] == "Reproduction":
                        # merge
                        slices[-2]["step_range"] = [slices[-2]["step_range"][0], end]
                        slices[-2]["content"] = traj[slices[-2]["step_range"][0]: end + 1]
                        slices.pop()
                    
                    else:
                        slices.append({
                            "process": cur_act,
                            "step_range": [start, end],
                            "content": traj[start: end + 1]
                        })
                    start = end + 1
                    cur_act = next_act
        return slices

    @staticmethod
    def extract_err(traj: list) -> list:
        err_list = []
        for idx in range(len(traj)):
            if traj[idx]['type'] == 'user' and traj[idx]["message"]["content"][0].get("is_error", False):
                assert len(traj[idx]["message"]["content"]) == 1, traj[idx]["message"]["content"]
                for begin in range(idx - 1, 0, -1):
                    if traj[begin]['type'] == 'user':
                        break
                err_list.append({
                    "step_range": [begin + 1, idx],
                    "content": traj[begin + 1: idx + 1]
                })
        return err_list

class RewardEstimator:
    class ParsedPatch(TypedDict):
        start: int
        end: int
        deleted_lines: list[str]
        added_lines: list[str]

    def __init__(self, 
                 container: Runtime, 
                 pred_patch: str, 
                 gold_patch: str):
        
        self.container: Runtime = container
        self.reproduction_file: str
        self.solution_patch: str
        self.reproduction_patch, self.solution_patch = self._reproduction_solution_patch_separator(pred_patch)
        self.gold_patch: str = gold_patch
        self.last_sim: float = 0.0
        self.last_diff: str = ""
        self.container.send_command("git stash; git reset --hard HEAD;")
        self.container.send_command("""perl -v &> /dev/null || (if command -v apt-get &> /dev/null; then apt-get update && apt-get install -y perl; elif command -v yum &> /dev/null; then yum install -y perl; elif command -v dnf &> /dev/null; then dnf install -y perl; elif command -v apk &> /dev/null; then apk add perl; elif command -v pacman &> /dev/null; then pacman -S --noconfirm perl; else echo "No supported package manager found" && exit 1; fi)""")

    @property
    def current_patch(self) -> str:
        return (self.container
                .send_command("git --no-pager diff HEAD")
                .output
                .replace("git --no-pager diff HEAD\n", ""))
    
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
        
        result = self.container.send_command(cmd)
        status = int(result.metadata.exit_code) == 0
        if not status:
            print(result.output)
        return status
    
    @staticmethod
    def _parse_patch(patch: str) -> dict[str, list[ParsedPatch]]:
        # use unidiff to get file_path : [(start1, end1, deleted_lines1, added_lines1), (start2, end2, deleted_lines2, added_lines2) ...] mapping
        # The added and deleted lines should not have + - and \n

        result: dict[str, list[RewardEstimator.ParsedPatch]] = {}
        
        try:
            patch_set = unidiff.PatchSet(patch)
        except Exception:
            # If patch parsing fails, return empty dict
            return result
        
        for patched_file in patch_set:
            file_path = patched_file.path
            patches: list[RewardEstimator.ParsedPatch] = []
            
            for hunk in patched_file:
                deleted_lines = []
                added_lines = []
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
                result[file_path] = sorted(patches, key = lambda x: x["start"])
        
        return result
    
    @staticmethod
    def localization(slice: TrajSlice) -> RewardReturnType:
        pass

    @staticmethod
    def reproduction_solution_patch_separator(patch: str) -> tuple[str, str]:
        '''
        extract the newly created reproduction.py file content (not diff) as reproduction_file
        discard newly added or deleted or binary files except reproduction.py
        retain other patches as solution_patch
        '''

        reproduction_patch = ""
        solution_patch_files: list[str] = []
        
        try:
            patch_set = unidiff.PatchSet(patch)
        except Exception:
            # If patch parsing fails, return empty strings
            return "", ""
        
        for patched_file in patch_set:
            # Check if this is reproduction.py and it's a newly added file
            if patched_file.path.endswith('reproduction.py') and patched_file.is_added_file:
                reproduction_patch = str(patched_file)
            
            # Skip deleted files and binary files
            elif patched_file.is_removed_file or patched_file.is_binary_file or patched_file.is_added_file:
                continue
            
            # Keep modified files as part of solution patch
            else:
                solution_patch_files.append(str(patched_file))
        
        solution_patch = ''.join(solution_patch_files)
        
        return reproduction_patch, solution_patch
    
    def reproduction(self, slice: TrajSlice) -> RewardReturnType:
        command = "python /testbed/reproduction.py"
        self.container.send_command("cat > /testbed/reproduction.py <<'REPRODUCTION_FILE'\n" + self.reproduction_file + "\nREPRODUCTION_FILE\n")
        pass

    def edit(self, slice: TrajSlice) -> RewardReturnType:
        pass

    def main(self, traj: ClaudeCodeTraj) -> list[RewardReturnType]:
        results: list[RewardReturnType] = []

        slices: list[TrajSlice] = TajectoryProcessor.split(traj)
        loc_slices: list[TrajSlice] = []
        repro_slices: ClaudeCodeTraj = []
        edit_slices: TrajSlice = []
        for slice in slices:
            if slice["process"] == "Localization":
                loc_slices.append(slice)
            if slice["process"] == "Reproduction":
                repro_slices.extend(slice["content"])
            if slice["process"] == "Edit":
                edit_slices.extend(slice["content"])

        reproduction_reward: RewardReturnType = self.reproduction(repro_slices)
        self.container.send_command("git stash; git reset --hard HEAD;")
        reproduction_reward: RewardReturnType = self.edit(edit_slices)
        return results


if __name__ == "__main__":
    with open("QwenCoder30B_django__django-16333_agent_side.txt") as f:
        traj = json.load(f)
    
    slices = TajectoryProcessor.split(traj)
    errs = TajectoryProcessor.extract_err(traj)

    with open("slice_exp_django-16333.json", "w") as f:
        json.dump(slices, f, indent = True)

    with open("error_steps_exp_django-16333.json", "w") as f:
        json.dump(errs, f, indent = True)