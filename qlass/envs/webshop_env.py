import re
import logging
from typing import Tuple

from qlass.envs import BaseEnv
from qlass.tasks import WebShopTask
from qlass.utils import prompt_with_icl, prompt_without_icl, State
from eval.webshop.web_agent_site.envs import WebAgentTextEnv


logger = logging.getLogger("agent_frame")


class WebShopEnv(BaseEnv):
    def __init__(
        self,
        task: WebShopTask,
        env: WebAgentTextEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: WebShopTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env
        
        self.state = State()
    
    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action: (.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
        except:
            observation = f"Observation: Invalid format. The input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = 0
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        try:
            observation, reward, done, info = self.env.step(action=action)
            observation = f"Observation:\n{observation}"
            # available_actions = self.env.get_available_actions()
            # observation = f"Observation:\n{observation}\n\nAvailable Actions:\n{available_actions}"
        except AssertionError:
            observation = 'Observation: Invalid action!'
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state
    
    def reset(self,num_icl_examples=1) -> Tuple[str, State]:
        self.state = State()
        self.env.reset(self.session_id)
        #available_actions = self.env.get_available_actions()
        #cur_task = f"{self.env.observation}\nAvailable Actions:\n{available_actions}"
        cur_task = self.env.observation
        if num_icl_examples > 0:
            observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, num_icl_examples)
        else:
            observation, messages = prompt_without_icl(self.instruction, cur_task)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
