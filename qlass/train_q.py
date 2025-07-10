# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from dataclasses import dataclass, field
import json
import math
import pathlib
from typing import Dict, Optional
import os
root_dir = os.getcwd()
from pathlib import Path
import sys
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from qlass.q_network import QNet
from qlass.q_trainer import Q_Trainer
from qlass.data_utils import rank0_print, preprocess
import math
import os
import torch

import transformers

import os
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )
    apply_sigmoid: bool = field(
        default=False, metadata={"help": "Apply sigmoid to the output of the q network"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


class SupervisedDataset_Q(Dataset):
    """Dataset for supervised fine-tuning qnet."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, model_path: str = None):
        super(SupervisedDataset_Q, self).__init__()

        rank0_print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, model_path, [example['label'] for example in raw_data])
        # save data_dict
        self.input_ids = data_dict["input_ids"].to("cuda")
        self.labels = data_dict["labels"].to("cuda")
        self.attention_mask = data_dict["attention_mask"].to("cuda")
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i].to("cuda"),
            labels=self.labels[i].to("cuda"),
            attention_mask=self.attention_mask[i].to("cuda"),
        )

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, model_path: str = None
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        SupervisedDataset_Q
    )
    rank0_print("Loading data...")
    train_json = json.load(open(data_args.data_path, "r"))
    
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, model_path=model_path)
    
    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, model_path=model_path)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    # torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float32)
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.environ['RANK'] = str(training_args.local_rank)

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    
    model = QNet(
        hidden_size=config.hidden_size,
        args=model_args,
        pad_token_id=tokenizer.pad_token_id,
        task_mode='eto',
        model_args=model_args,
        training_args=training_args,
    ).to("cuda")
    

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, model_path=model_args.model_name_or_path)
    
    trainer = Q_Trainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        **data_module,
    )
    # Train
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    train()

