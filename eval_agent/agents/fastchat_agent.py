import json
import time
import logging
from typing import List, Dict, Union, Any
import requests
from fastchat_deprecated.model.model_adapter import get_conversation_template
from requests.exceptions import Timeout, ConnectionError

from .base import LMAgent

logger = logging.getLogger("agent_frame")


def _add_to_set(s, new_stop):
    if not s:
        return
    if isinstance(s, str):
        new_stop.add(s)
    else:
        new_stop.update(s)


class FastChatAgent(LMAgent):
    """This agent is a test agent, which does nothing. (return empty string for each action)"""

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.controller_address = config["controller_address"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 0)
        self.top_k = config.get("top_k", -1)
        self.get_choice_rank = config.get("get_choice_rank", 0)
        self.num_samples = config.get("num_samples", 2)

    def __call__(self, messages: List[dict],get_choice_rank:int=0) -> str:
        # import ipdb; ipdb.set_trace()
        # if get_choice_rank != -1:
        self.get_choice_rank = get_choice_rank
        controller_addr = self.controller_address
        worker_addr = controller_addr
        if worker_addr == "":
            raise ValueError
        gen_params = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "echo": False,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "get_choice_rank":0,
            "num_samples": 2,
        }
        conv = get_conversation_template(self.model_name)
        for history_item in messages:
            role = history_item["role"]
            content = history_item["content"]
            if role == "user":
                conv.append_message(conv.roles[0], content)
            elif role == "assistant":
                conv.append_message(conv.roles[1], content)
            else:
                raise ValueError(f"Unknown role: {role}")
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        new_stop = set()
        _add_to_set(self.stop_words, new_stop)
        _add_to_set(conv.stop_str, new_stop)
        gen_params.update(
            {
                "prompt": prompt,
                "stop": list(new_stop),
                "stop_token_ids": conv.stop_token_ids,
            }
        )
        headers = {"User-Agent": "FastChat Client"}
        for _ in range(3):
            # import ipdb; ipdb.set_trace()
            try:
                response = requests.post(
                    controller_addr + "/worker_generate_stream",
                    headers=headers,
                    json=gen_params,
                    stream=True,
                    timeout=120,
                )
                text = ""
                for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
                    if line:
                        data = json.loads(line)
                        if data["error_code"] != 0:
                            print(f"Error code: {data['error_code']}")
                            assert False, data["text"]
                        text = data["text"]
                return text
            # if timeout or connection error, retry
            # except Timeout:
            #     print("Timeout, retrying...")
            # except ConnectionError:
            #     print("Connection error, retrying...")
            except Exception as e:
                print(f"Error when agent try to continue the conversation: {e},retrying...")
            time.sleep(5)
        # else:
        raise Exception("Timeout after 3 retries.")
