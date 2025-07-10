
import torch
import torch.distributed as dist
import os
from typing import Dict, List, Tuple
from enum import Enum, auto
import transformers
from dataclasses import dataclass
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

# Borrored from https://github.com/sgl-project/sglang/blob/main/python/sglang/lang/chat_template.py
class ChatTemplateStyle(Enum):
    PLAIN = auto()
    LLAMA2 = auto()

@dataclass
class ChatTemplate:
    name: str
    default_system_prompt: str
    role_prefix_and_suffix: Dict[str, Tuple[str, str]]
    stop_str: List[str] = ()
    image_token: str = "<image>"
    style: ChatTemplateStyle = ChatTemplateStyle.PLAIN

    def get_prefix_and_suffix(
        self, role: str, hist_messages: List[Dict]
    ) -> Tuple[str, str]:
        prefix, suffix = self.role_prefix_and_suffix.get(role, ("", ""))

        if self.style == ChatTemplateStyle.LLAMA2:
            if role == "system" and not hist_messages:
                user_prefix, _ = self.role_prefix_and_suffix.get("user", ("", ""))
                system_prefix, system_suffix = self.role_prefix_and_suffix.get(
                    "system", ("", "")
                )
                return (user_prefix + system_prefix, system_suffix)
            elif (
                role == "user"
                and len(hist_messages) == 1
                and hist_messages[0]["content"] is not None
            ):
                return ("", suffix)

        return prefix, suffix

    def get_prompt(self, messages: List[Dict]) -> str:
        prompt = ""
        for i, message in enumerate(messages):
            role, content = message["role"], message["content"]
            if role == "system" and content is None:
                content = self.default_system_prompt
                if content is None:
                    continue

            prefix, suffix = self.get_prefix_and_suffix(role, messages[:i])

            if role == "assistant" and content is None and i == len(messages) - 1:
                prompt += prefix
            else:
                prompt += f"{prefix}{content}{suffix}"

        return prompt


def get_chat_template(model_path: str) -> ChatTemplate:

    if "llama-2" in model_path.lower() and "chat" in model_path.lower():
        return ChatTemplate(
            name="llama-2-chat",
            default_system_prompt=None,
            role_prefix_and_suffix={
                "system": ("<<SYS>>\n", "\n<</SYS>>\n\n"),
                "user": ("[INST] ", " [/INST]"),
                "assistant": ("", " </s><s>"),
            },
            style=ChatTemplateStyle.LLAMA2,
        )
    
    if "llama-3" in model_path.lower() and "instruct" in model_path.lower():
        return ChatTemplate(
            name="llama-3-instruct",
            default_system_prompt=None,
            role_prefix_and_suffix={
                "system": (
                    "<|start_header_id|>system<|end_header_id|>\n\n",
                    "<|eot_id|>",
                ),
                "user": (
                    "<|start_header_id|>user<|end_header_id|>\n\n",
                    "<|eot_id|>",
                ),
                "assistant": (
                    "<|start_header_id|>assistant<|end_header_id|>\n\n",
                    "<|eot_id|>",
                ),
            },
            stop_str=("<|eot_id|>",),
            image_token="<|image|>",
        )
    
    assert False, f"ChatTemplate not implemented for {model_path}"


def rank0_print(message):
    # Assuming rank 0 is determined by some condition, e.g., an environment variable or a global variable
    rank = int(os.getenv('RANK', '0'))
    if rank == 0:
        print(message)


def check_length_after_masking(
        conversation: str, target: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, cur_len: int, total_len: int, convs: List[str],
        debug: bool = False
    ):

    if debug:  # Inspect and check the correctness of masking
        z = target.clone()
        z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        rank0_print(conversation)
        rank0_print(tokenizer.decode(z))
        exit()

    if cur_len < tokenizer.model_max_length:
        if cur_len != total_len:
            import ipdb; ipdb.set_trace()
            target[:] = IGNORE_TOKEN_ID
            rank0_print(
                f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                f" #turn = {len(convs) - 1}. (ignored)"
            )
        # else:
        #     import ipdb; ipdb.set_trace()
        #     print("total_len==cur_len")


def mask_non_assistant_tokens_llama_2_chat(conversation: str, target: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer):
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    turn_sep = " </s><s>"
    conv_sep = " [/INST]"
    turns = conversation.split(turn_sep)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID

    for i, turn in enumerate(turns):
        if turn == "":
            break

        # remove <s>
        turn_len = len(tokenizer(turn).input_ids) - 1
        # fix: inconsistent tokenization by llama tokenizer #3006
        # turn_len = len(tokenizer(turn, add_special_tokens=False).input_ids)
        parts = turn.split(conv_sep)

        if len(parts) != 2:
            break
        parts[0] += conv_sep
        
        # remove <s> and the "_" in the end
        instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        # Ignore the user instructions
        target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID

        # add the length of turn sep
        cur_len += turn_len + 3

    target[cur_len:] = IGNORE_TOKEN_ID

    check_length_after_masking(conversation, target, tokenizer, cur_len, total_len, turns, debug=False)
    return target


def mask_non_assistant_tokens_llama_3_instruct(conversation: str, target: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer):
    total_len = int(target.ne(tokenizer.pad_token_id).sum())
    conv_sep = "<|eot_id|>"
    assistant_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    convs = conversation.split(conv_sep)
    cur_len = 1
    target[:cur_len] = IGNORE_TOKEN_ID
    for conv in convs:
        conv += conv_sep
        turn_len = len(tokenizer(conv).input_ids) - 1
        if assistant_prefix not in conv:
            target[cur_len : cur_len + turn_len] = IGNORE_TOKEN_ID
        else:
            prefix_len = len(tokenizer(assistant_prefix).input_ids) - 1
            target[cur_len : cur_len + prefix_len] = IGNORE_TOKEN_ID

        cur_len += turn_len
    
    target[cur_len:] = IGNORE_TOKEN_ID
    cur_len -= 1

    check_length_after_masking(conversation, target, tokenizer, cur_len, total_len, convs, debug=False)
    return target


def mask_non_assistant_tokens(conversation: str, target: torch.Tensor, tokenizer: transformers.PreTrainedTokenizer, chat: ChatTemplate):
    if chat.name == "llama-2-chat":
        return mask_non_assistant_tokens_llama_2_chat(conversation, target, tokenizer)
    elif chat.name == "llama-3-instruct":
        return mask_non_assistant_tokens_llama_3_instruct(conversation, target, tokenizer)
    
    assert False, f"mask_non_assistant_tokens not implemented for {chat.name}"

def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    model_path: str,
    rewards: List[float] = None,
) -> Dict:
    chat:ChatTemplate = get_chat_template(model_path)
    roles = {"human": "user", "gpt": "assistant"}
    roles_list = ["user", "assistant"]
    max_length = 0

    # Apply prompt templates
    conversations = []

    # rank0_print(f"Processing {len(sources)} conversations...")

    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != "user":
            # Skip the first one if it is not from human
            source = source[1:]

        messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == roles_list[j % 2], f"{i}"
            messages.append(
                {"role": role, "content": sentence["value"]}
            )
        conversations.append(chat.get_prompt(messages))

    # Tokenize conversations
    # rank0_print("Tokenizing conversations...")

    # check whether dist is initialized
    if not dist.is_initialized():
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    else:
        # Tokenize conversations only at rank 0 and then using all_gather to broadcast the tokenized conversations
        rank = int(os.getenv('RANK', '0'))
        if rank == 0:
            input_ids = tokenizer(
                conversations,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ).input_ids
            # Put input_ids to the current cuda device
            input_ids = input_ids.to(torch.device("cuda:0"))
        else:
            input_ids = torch.empty(
                (len(conversations), tokenizer.model_max_length),
                dtype=torch.long,
                device=torch.device(f"cuda:{rank}"),
            )

        dist.broadcast(input_ids, src=0)

        # Cast input_ids back to the cpu
        input_ids = input_ids.to(torch.device("cpu"))

    # When no rewards are provided, we need to mask the non-assistant tokens to do SFT
    if rewards is None:
        targets = input_ids.clone()
        max_length = max(max_length, input_ids.size(1))

        # Mask targets. Only compute loss on the assistant outputs.
        for conversation, target in zip(conversations, targets):
            target = mask_non_assistant_tokens(conversation, target, tokenizer, chat)
        
        rank0_print(f"Max length seq length: {max_length}")
    else:
        targets = torch.Tensor(rewards)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

if __name__ == '__main__':
    import json
    model_path = '/mnt/model/Llama-3.2-1B-instruct'
    data_path = '/mnt/data/train/alfworld/alfworld_sft.json'
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=None,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        # use_fast=True,
        trust_remote_code=False,
    )
    train_json = json.load(open(data_path, "r"))
    sources = [example["conversations"] for example in train_json]
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.unk_token = tokenizer.pad_token
    preprocess(sources, tokenizer, model_path)