import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore
import qlass.tasks as tasks
import qlass.agents as agents
import qlass.envs as envs
from qlass.utils import State


logger = logging.getLogger("agent_frame")

N_DEBUG_EXAMPLES = 5

def debug_logging_info(msg: str, debug: bool):
    if debug:
        logger.info(msg)
    return
    
def interactive_loop(
    args: argparse.Namespace,
    task: tasks.Task,
    agent: agents.LMAgent,
    env_config: Dict[str, Any],
    debug: bool = False,
) -> State:
    debug_logging_info(f"Loading environment: {env_config['env_class']}", debug)
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    if args.force_first:
        env.icl_format = 'first'
        
    # reset the environment and set the prompt
    observation, state = env.reset(args.num_icl_examples)


    init_msg = observation

    debug_logging_info(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}", debug)

    cur_step = 1
    # import ipdb;ipdb.set_trace()
    while not state.finished:
        debug_logging_info(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n", debug)
        cur_step += 1
        # agent act
        try:
            llm_output: str = agent(state.history)
            debug_logging_info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n", debug)
        except Exception as e:
            logger.info(f"Agent failed with error: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = f"Agent failed with error: {e}"
            break
        # environment step
        observation, state = env.step(llm_output)
        if not state.finished:
            # color the observation in blue
            debug_logging_info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n", debug)
        if state.finished:
            break

    if state.reward is not None:
        debug_logging_info(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}", debug)
    else:
        debug_logging_info(f"Task finished in {state.steps} steps. Success: {state.success}", debug)
        
    return state


def main(args: argparse.Namespace):
    # set_seed(42)
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
    
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name

    output_path = os.path.join("outputs4", agent_config['config']['model_name'].replace('/', '_'), args.exp_config+args.exp_name+args.split+str(args.num_icl_examples)+'shot')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'WebShopEnv':
        from eval.webshop.web_agent_site.envs import WebAgentTextEnv
        env_config['env'] = WebAgentTextEnv(observation_mode="text", human_goals=True)
    elif env_config['env_class'] == 'SciWorldEnv':
        from scienceworld import ScienceWorldEnv
        from eval_agent.utils.replace_sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config['env'] = ScienceWorldEnv("", serverPath=os.path.join(os.getcwd(), env_config['env_jar_path']), envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )

    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        logger.info(f"Existing output file found. {len(done_task_id)} tasks done.")


    if len(done_task_id) == n_tasks:
        logger.info("All tasks done. Exiting.")
        return

    # # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks
    n = 0
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(
                args, task, agent, env_config, args.debug
            )

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
            n+=1
            if args.debug and n>=N_DEBUG_EXAMPLES:
                break
        pbar.close()
    
    logger.warning("All tasks done.")
    # logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(int(state.success==1.))
    print(f"all traj len is {len(state_list)}")
    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="webshop",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./qlass/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=False,
        help="Model name. It will override the 'model_name' in agent_config"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--force_first",
        action="store_true",
        help="Whether to use first in icl_format.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    parser.add_argument(
        "--num_icl_examples",
        type=int,
        default=1,
        help="Number of ICL examples in evaluation prompt.")
    parser.add_argument(
        "--notes",
        type=str,
        default='top_k=-1',
        help="evaluation setting notes")
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)