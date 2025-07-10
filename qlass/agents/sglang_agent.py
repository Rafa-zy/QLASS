import json
import time
import logging
from typing import List
import requests
from qlass.data_utils import get_chat_template

from .base import LMAgent

logger = logging.getLogger("agent_frame")


class SGLangAgent(LMAgent):

    def __init__(
        self,
        config
    ) -> None:
        super().__init__(config)
        self.server_address = config["server_address"]
        self.model_name = config["model_name"]
        self.temperature = config.get("temperature", 0)
        self.max_new_tokens = config.get("max_new_tokens", 512)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)
        self.get_choice_rank = config.get("get_choice_rank", 0)
        self.num_samples = config.get("num_samples", 1)

    def __call__(self, messages: List[dict],get_choice_rank:int=0) -> str:
        # import ipdb; ipdb.set_trace()
        # if get_choice_rank != -1:
        self.get_choice_rank = get_choice_rank
        server_addr = self.server_address
        if server_addr == "":
            raise ValueError
        sampling_params = {
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "n": self.num_samples,
        }
        chat = get_chat_template(self.model_name)
        prompt = chat.get_prompt(messages + [{"role": "assistant", "content": None}])
        for _ in range(3):
            # import ipdb; ipdb.set_trace()
            try:
                response = requests.post(
                    server_addr + "/generate",
                    json = {
                        "text": prompt,
                        "sampling_params": sampling_params,
                        "stream": True,
                    },
                )
                text = ""
                for chunk in response.iter_lines(decode_unicode=False):
                    chunk = chunk.decode("utf-8")
                    if chunk and chunk.startswith("data:"):
                        if chunk == "data: [DONE]":
                            break
                        data = json.loads(chunk[5:].strip("\n"))
                        text = data["text"].strip()
                return text

            except Exception as e:
                print(f"Error when agent try to continue the conversation: {e},retrying...")
            time.sleep(5)
        return ""
