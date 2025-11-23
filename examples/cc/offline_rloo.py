# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl[vllm]",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify num2words==0.5.14 peft trackio vllm
export TRACKIO_PROJECT="RLOO-NuminaMath-TIR"
accelerate launch --config_file examples/accelerate_configs/deepspeed_zero3.yaml examples/scripts/rloo.py
"""

import argparse
import os
import warnings
from dataclasses import dataclass

import datasets
import torch
from offline_rloo_trainer import OfflineRLOOTrainer
from peft import LoraConfig  # , PromptEncoderConfig
from transformers import TrainingArguments
from trl import RLOOConfig

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def prepare_offline_dataset(dataset_path):
    ds = datasets.load_from_disk(dataset_path)
    train_dataset, eval_dataset = ds["train"], ds["test"]

    return train_dataset, eval_dataset


def parse_args():
    # Args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset_dir", type=str, default="data/dataset/personalized_datasets/sympy_sympy")
    parser.add_argument("--output_dir", type=str, default="Qwen3-0.6B-RLOO")
    args = parser.parse_args()
    return args


@dataclass
class OfflineRLOOConfig(RLOOConfig):
    def __post_init__(self):
        """
        This overrides the original __post_init__ to avoid num_generation checking
        """
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        _DEPRECATED_PARAMS = {
            "rloo_k": "num_generations",
            "cliprange": "epsilon",
            "kl_coef": "beta",
            "exp_name": "run_name",
            "normalize_reward": "normalize_advantages",
            "num_ppo_epochs": "num_iterations",
            "num_mini_batches": "steps_per_generation",
            "total_episodes": "max_steps",
            "response_length": "max_completion_length",
        }

        _REMOVED_PARAMS = {
            "token_level_kl",
            "dataset_num_proc",
            "local_rollout_forward_batch_size",
            "num_sample_generations",
            "stop_token",
            "stop_token_id",
            "missing_eos_penalty",
        }

        # Check for deprecated parameters and issue warnings
        for old_param, new_param in _DEPRECATED_PARAMS.items():
            if getattr(self, old_param) is not None:
                old_value = getattr(self, old_param)
                if old_param == "total_episodes":
                    old_value = old_value // self.gradient_accumulation_steps
                warnings.warn(
                    f"Parameter '{old_param}' is deprecated and will be removed in version 0.25.0. Please use "
                    f"'{new_param}' instead. We are setting {new_param}={old_value}"
                )
                # Set the new parameter with the old value
                setattr(self, new_param, old_value)
                # Clear the deprecated parameter
                setattr(self, old_param, None)

        for removed_param in _REMOVED_PARAMS:
            if hasattr(self, removed_param) and getattr(self, removed_param) is not None:
                warnings.warn(
                    f"Parameter '{removed_param}' is deprecated and will be removed in version 0.25.0. Please refer "
                    "to the migration guide: https://huggingface.co/docs/trl/en/rloo_trainer##migration-guide-from-the-old-implementation-021-and-below"
                )

        TrainingArguments.__post_init__(self)

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (self.per_device_train_batch_size * num_processes)
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )


def main():
    args = parse_args()

    training_args = OfflineRLOOConfig(
        output_dir=args.output_dir,
        model_init_kwargs={"dtype": torch.bfloat16},
        learning_rate=1e-5,
        # gradient_checkpointing=False,
        gradient_checkpointing_kwargs=dict(use_reentrant=True),
        log_completions=True,
        num_completions_to_print=2,
        max_prompt_length=2048,
        max_completion_length=1024,
        num_train_epochs=1.0,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        steps_per_generation=1,
        num_generations=1,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.5,
        run_name="Qwen3-0.6B-RLOO-NuminaMath-TIR",
        report_to="trackio",  # [trackio, wandb]
    )

    # Datasets
    train_dataset, eval_dataset = prepare_offline_dataset(args.dataset_dir)

    def simple_reward(reward, completion_ids, gold_completion_ids, **kwargs):
        rewards = []
        for _reward, completion, gold_completion in zip(reward, completion_ids, gold_completion_ids):
            _reward = float(completion == gold_completion) * _reward
            rewards.append(_reward)
        return rewards

    # Training
    peft_config = LoraConfig()
    # prompt_tuning_init_text = "Help the user to fix bugs in this repo.\n"
    # peft_config = PromptEncoderConfig(
    #     task_type="CAUSAL_LM",
    #     num_virtual_tokens=20, encoder_hidden_size=128
    # )

    trainer = OfflineRLOOTrainer(
        model=args.model_name_or_path,
        args=training_args,
        reward_funcs=[simple_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    # trainer.push_to_hub(dataset_name="AI-MO/NuminaMath-TIR")
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
