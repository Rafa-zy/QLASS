import re
import os
import yaml
import logging
import textworld
from typing import Tuple

from qlass.envs import BaseEnv
from qlass.tasks import AlfWorldTask
from qlass.utils import State, prompt_with_icl, prompt_without_icl

from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos, AlfredExpert


logger = logging.getLogger("agent_frame")


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]
    return ob


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        # self.env = task.env
        self.env = self._load_single_task_env(self.task.game_file)
        self.state = State()
    
    def _load_single_task_env(self, gamefile: str):
        alfworld_data_path = "eval_agent/data/alfworld"
        with open(os.path.join(alfworld_data_path, "base_config.yaml")) as f:
            config = yaml.safe_load(f)
        domain_randomization = config["env"]["domain_randomization"]
        if self.task.split != "train":
            domain_randomization = False

        alfred_demangler = AlfredDemangler(shuffle=domain_randomization)
        wrappers = [alfred_demangler, AlfredInfos]

        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        expert_type = config["env"]["expert_type"]
        training_method = config["general"]["training_method"]

        if training_method == "dqn":
            max_nb_steps_per_episode = config["rl"]["training"]["max_nb_steps_per_episode"]
        elif training_method == "dagger":
            max_nb_steps_per_episode = config["dagger"]["training"]["max_nb_steps_per_episode"]
            expert_plan = True if self.task.split == "train" else False
            if expert_plan:
                wrappers.append(AlfredExpert(expert_type))
                request_infos.extras.append("expert_plan")
        
        env_id = textworld.gym.register_games([gamefile], request_infos,
            batch_size=1,
            asynchronous=True,
            max_episode_steps=max_nb_steps_per_episode,
            wrappers=wrappers
        )
        # Launch Gym environment.
        env = textworld.gym.make(env_id)
        return env

    def parse_action(self, llm_output: str) -> str:
        llm_output = llm_output.strip()
        pattern = re.compile(r"Action:\s?(.*)", re.DOTALL)
        action = re.findall(pattern, llm_output)[0]
        assert action is not None
        return action
    
    def conduct_action(self, action: str):
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        # import pdb; pdb.set_trace()
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)
        except Exception as e:
            # logger.debug(f"Agent failed with error: {e}")
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state


        observation = f"Observation: {observation}"
        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward

        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self,num_icl_examples=1) -> Tuple[str, State]:
        self.state = State()
        self.state.error = self.task.game_file
        cur_task = self.task.observation
        #observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, num_icl_examples)
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
        self.env.reset()
        return observation, self.state
