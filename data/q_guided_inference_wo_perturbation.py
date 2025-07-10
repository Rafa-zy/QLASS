# Semantic perturbation functions on original trajectories
# from openai import OpenAI
import openai
import json
import os
import argparse
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import sys
import os
root_dir = os.getcwd()
from pathlib import Path
sys.path.append(root_dir)
sys.path.append((Path(root_dir).parent))
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import pickle
from itertools import islice
import random
import copy
from model.q_network import QNet
from model.orm import OutcomeRewardModel

try:
    from eval.hotpotqa.zeno_build.models import lm_config
except:
    from zeno_build.models import lm_config
from data.fast_prompt import *
import re
import random
import torch
import gym
import numpy as np
from rich import print
from explore_agent_sft import TreeNode

from eval.utils import generate_completions, load_hf_lm_and_tokenizer

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
from typing import List, Dict, Any
import logging
from colorama import Fore
from eval_agent.prompt.templates import *
from eval_agent.prompt.instructions import *

logger = logging.getLogger("agent_frame")
openai.api_key = os.environ["OPENAI_API_KEY"]
model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-3.5-turbo")
eval_model_config = lm_config.LMConfig(provider="openai_chat", model="gpt-4-1106-preview")
engine = 'gpt-3.5-turbo'
perturb_prompt = """
Please paraphrase the following text:
"""
root_dir = '/local2/xingcheng/Q-Agent/'

# def perturb_messages(instruction,messages):

#     message = messages[0]['content'].split('here is the task.\nWebShop [SEP] Instruction: [SEP] ')[-1].split(' [SEP] Search')[0].strip()
#     predictions = generate_from_openai_chat_completion(
#             full_contexts=[chat_prompt.ChatMessages(
#                 [{"role": "user", "content": f"Paraphrase the task:\n\n{message}\n\n"}]
#             )],
#             model_config=model_config,
#             temperature=0.7+0.1*random.random(),
#             max_tokens=200,
#             tqdm=False
#         )
#     # print("predictions:", predictions)
#     task = f"WebShop [SEP] Instruction: [SEP] {predictions[0].strip()} [SEP] Search"
#     messages[0]['content'] = PROMPT_WITHOUT_ICL_TEMPLATE.format(instruction=instruction, task=task)
#     print("messages[0]:", messages[0])

#     return messages

@torch.no_grad()
def evaluate_trajs_qnet_v2(model, tokenizer, new_state, batch_size=1, disable_tqdm=False):
    q_values = []
    sources = [new_state.to_dict()['conversations']]
    if not disable_tqdm:
        progress = tqdm.tqdm(total=len(sources), desc="Evaluating Trajectories")
    device = next(model.parameters()).device
    
    from fastchat_deprecated.train.train_q import preprocess_Q
    data_dict = preprocess_Q(sources, tokenizer,'/local2/xingcheng/model/Llama-2-7b-chat-hf/',[0. for i in range(len(sources)) ] )
    
    for i in range(0, len(sources), batch_size):
        # batch_trajs = sources[i:i + batch_size]
        # tokenized_trajs = tokenizer(batch_trajs, padding="longest", return_tensors="pt", add_special_tokens=True)
        # batch_input_ids = tokenized_trajs.input_ids.to(device)
        # attention_mask = tokenized_trajs.attention_mask.to(device)
        batch_input_ids = data_dict['input_ids'][i:i + batch_size].to(device)
        attention_mask = data_dict['attention_mask'][i:i + batch_size].to(device)
        try:
            # QNet forward pass
            q_output = model(batch_input_ids, attention_mask=attention_mask)
            # Assuming QNet returns a single scalar per example as Q-value
            # import pdb; pdb.set_trace()
            batch_q_values = q_output[:,-1].squeeze().tolist()  # Ensure it's a list
            if isinstance(batch_q_values, float):
                batch_q_values = [batch_q_values]
        except Exception as e:
            print(f"Error when evaluating trajectories for batch:{e}")
            # print(batch_trajs)
            # print("Error message:")
            # print(e)
            # # Use a default Q value (e.g., -inf or a large negative number to indicate failure)
            # batch_q_values = [float('-inf')] * len(batch_trajs)

        q_values.extend(batch_q_values)

        if not disable_tqdm:
            progress.update(len(batch_trajs))

    return q_values

# need to deploy at local env.
def draw_graph_v2(G, pos, idx=0):
    # Use Graphviz layout to draw the tree
    pos = graphviz_layout(G, prog='dot')  # 'dot' engine gives a top-down layout
    labels = nx.get_node_attributes(G, 'label')
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=3000, node_color='skyblue', font_size=9, edge_color='gray', font_weight='bold')
    plt.savefig(f"tree_{idx}.png")
    plt.show()
    
def draw_graph(G, pos, idx=0):
    labels = {node: G.nodes[node]['label'] for node in G}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=2500, node_color='skyblue', font_size=9, font_weight='bold', edge_color='gray')
    plt.savefig(f"tree_{idx}.png")
    plt.show()
    



def collect_and_print_stats(node):
    """ Recursively collect stats from the tree and print them. """
    def recurse_collect(node):
        rewards = [node.reward] if node.reward is not None else []
        q_values = [node.q_value] if node.q_value is not None else []
        
        for child in node.children:
            child_rewards, child_q_values = recurse_collect(child)
            rewards.extend(child_rewards)
            q_values.extend(child_q_values)
        return rewards, q_values
    
    rewards, q_values = recurse_collect(node)
    
    if rewards:
        average_reward = sum(rewards) / len(rewards)
        min_reward = min(rewards)
        max_reward = max(rewards)
    else:
        average_reward = min_reward = max_reward = None

    if q_values:
        average_q_value = sum(q_values) / len(q_values)
        min_q_value = min(q_values)
        max_q_value = max(q_values)
    else:
        average_q_value = min_q_value = max_q_value = None

    print(f"Average Reward: {average_reward}")
    print(f"Min Reward: {min_reward}")
    print(f"Max Reward: {max_reward}")
    print(f"Average Q Value: {average_q_value}")
    print(f"Min Q Value: {min_q_value}")
    print(f"Max Q Value: {max_q_value}")

def find_top_k_trajectories(root, k=3):
    """ This function finds the top k trajectories based on the sum of Q-values. """
    # Store the sum of Q-values for each trajectory
    trajectory_sums = {}

    def calculate_q_sum(node, current_path, current_sum):
        current_sum += node.q_value
        if not node.children:
            # If no children, it means the path ends here, record the sum
            trajectory_sums[tuple(current_path)] = current_sum
        else:
            # Otherwise, continue down each child
            for child in node.children:
                calculate_q_sum(child, current_path + [child], current_sum)

    # Start the recursive function from root
    calculate_q_sum(root, [root], 0)

    # Sort the trajectories by their Q-value sums, in descending order
    sorted_trajectories = sorted(trajectory_sums.items(), key=lambda item: item[1], reverse=True)

    # Return the top k trajectories
    return sorted_trajectories[:k]

def verify_conversations(conversations):
    if not conversations:
        return False  # Return False if the list is empty
    
    # Check if the list starts with a human message and alternates properly
    if conversations[0]['from'] != 'human':
        return False  # First message must be from the human
    
    # Verify alternating roles and that it ends with gpt
    expected_from = 'human'
    for message in conversations:
        if message['from'] != expected_from:
            return False  # 'from' field does not match the expected role
        # Switch the expected role for the next iteration
        expected_from = 'gpt' if expected_from == 'human' else 'human'
    
    # Check if the last message is from gpt
    if conversations[-1]['from'] != 'user':
        return False  # Last message must be from gpt
    
    return True  # All checks passed


# GEN_FUNC = interactive_loop_for_explore
# RECURSIVE_MODE = True
# N_SFT_EXAMPLES = 1000

MAX_TURNS={"webshop":5,"sciworld":15,"alfworld":10}
N_SAMPLE = 10
# N_TRAJS = 1

EPSILON = 0.1
TOPK = 2


def main(args):
    import wandb
    import omegaconf
    #wandb_config = omegaconf.OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    wandb.init(project="qagent-q-guided-eval",
               entity="tangyao2020",
               name=args.exp_name,
               config=args,
               settings=wandb.Settings(
                   start_method="thread",
                   _disable_stats=True,
               ),
               mode="online",
            #    notes=args.notes,
               )
    total_examples = []
    total_trees = []
    successful_trajs = []
    token_count = 0
    
    args.model_name_or_path = args.qnet_path
    args.low_cpu_mem_usage = False
    args.use_flash_attn = True
    qnet = QNet.from_pretrained(args.qnet_path,None,args)
    qnet = qnet.to("cuda")
    qnet.device = torch.device("cuda")
    
    print("loaded qnet successfully")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('/local2/xingcheng/model/Llama-2-7b-chat-hf/',
                                                model_max_length=4096,
                                                use_fast=False)
    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    random.seed(42)

    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)
        
    if args.model_name is not None:
        agent_config['config']['model_name'] = args.model_name
        agent_config['config']['batch_size'] = args.eval_batch_size
        
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
        #env_config['env'] = ScienceWorldEnv("", serverPath=f"{root_dir}/envs/scienceworld/scienceworld.jar", envStepLimit=200)

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.slice_num, args.slice_id) 
    
    # initialize the agent
    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(
        agent_config["config"]
    )
    # Initialize output files
    # from datetime import datetime
    # today = datetime.today().date()
    # traj_file = f'{root_dir}/data/train/guided_explore/{today}/guided_explore_{args.exp_config}_{args.sample_mode}_{args.model_name}_{args.split}_slice{args.slice_id}of{args.slice_num}_{args.num_icl_examples}shot.jsonl'
    # tree_file = f'{root_dir}/data/train/guided_explore/{today}/guided_explore_{args.exp_config}_{args.sample_mode}_{args.model_name}_{args.split}_slice{args.slice_id}of{args.slice_num}_{args.num_icl_examples}shot.pkl'
    # os.makedirs(os.path.dirname(traj_file), exist_ok=True)
    # os.makedirs(os.path.dirname(tree_file), exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    traj_file = args.output_dir+f"{args.slice_id}of{args.slice_num}_slices_{args.sample_mode}_traj.jsonl"
    tree_file =args.output_dir+f"{args.slice_id}of{args.slice_num}_slices_{args.sample_mode}_tree.pkl"
    # output_dir = os.path.dirname(traj_file)
    

    done_task_id = []
    with logging_redirect_tqdm():
        pbar = tqdm(total=n_tasks)
        for i, task in enumerate(all_tasks):
            if args.debug and i==2:
                break
            
            ds = args.exp_config
            all_trajs = []
            print(f"Task {i}")
            
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue
            
            env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
            env.icl_format = 'first'
            env.max_steps = MAX_TURNS[args.exp_config]
            init_msg, state = env.reset(num_icl_examples=args.num_icl_examples) 

            root = TreeNode(state=init_msg, action='No Action (Root)', reward=state.reward) 
            
            instruction_path = f"{root_dir}/eval_agent/prompt/instructions/{args.exp_config}_inst.txt"
            with open(instruction_path) as f:
                instruction = f.read()
            for _ in range(args.n_trajs):
                init_msg , cur_traj_state = env.reset(num_icl_examples=args.num_icl_examples) 
                for n_turn in range(MAX_TURNS[args.exp_config]):
                    new_state_candidates = []
                    if n_turn==0:
                        current_node=root
                    for idx in range(args.best_of_N):
                        # import ipdb; ipdb.set_trace()
                        observation, state = env.reset(num_icl_examples=args.num_icl_examples) 
                        if n_turn >0:
                            for i in range(1,n_turn+1):
                                action = cur_traj_state.history[i*2-1]['content']
                                observation, state = env.step(action)
        
                        assert state.history==cur_traj_state.history
                    
                        # if not idx:
                        cur_state_history = cur_traj_state.history
                        # else:
                        #     cur_state_history = perturb_messages(instruction,copy.deepcopy(cur_traj_state.history))
                            
                        action = agent(cur_state_history)
                        observation, new_state = env.step(action)
                        if new_state.finished:
                            new_state_candidates.append((new_state,new_state.reward))
                        else:
                            Q_value = evaluate_trajs_qnet_v2(qnet, tokenizer,new_state, batch_size=1, disable_tqdm=True)[0]
                            new_state_candidates.append((new_state,Q_value))
                            
                    # Select the best action from candidates based on the Q-values 
                    if args.sample_mode == 'epsilon_greedy':
                        if random.random() < EPSILON :  # Noted that this is not leakage, just a stopping signal from the model itself
                            # Explore: randomly select a trajectory
                            selected_traj = random.choice(new_state_candidates)
                        else:
                            # Exploit: select the best trajectory based on Q values
                            selected_traj = max(new_state_candidates, key=lambda x: x[1])
                    elif args.sample_mode == 'bon':
                        selected_traj = max(new_state_candidates, key=lambda x: x[1])
                    else:
                        raise NotImplementedError(f"We do not support the sample mode: {args.sample_mode}")

                    new_state, reward = selected_traj
                    new_node = TreeNode(state=new_state.to_dict()['conversations'][:-2], action=new_state.to_dict()['conversations'][-2], reward= reward)
                    assert isinstance(new_state.to_dict()['conversations'][-2], dict) and new_state.to_dict()['conversations'][-2]['from']=='gpt'
                    current_node.add_child(new_node)
                    current_node = new_node
                    cur_traj_state = new_state
                    if cur_traj_state.finished:
                        break
                all_trajs.append(
                    {
                        'dataset': ds,
                        'id': task.task_id,
                        'conversations': cur_traj_state.to_dict()['conversations'],
                        'reward': cur_traj_state.reward,
                        'success': cur_traj_state.success,
                    }
                )
            collect_and_print_stats(root)
            # max_reward_traj = max(all_trajs, key=lambda traj: traj['reward'])
            # for here, we only save the best trajectory for self-training, but this step needs to be modified for the DPO version to get preference-pair
            # total_examples += [max_reward_traj]
            total_examples += all_trajs
            id = task.task_id
            total_trees.append({'dataset':ds, 'id': id, 'tree': root})
            
            # if id % 2 == 0:
            with open(traj_file, 'w') as f:
                json.dump(total_examples, f)
            
            # save as pickle
            with open(tree_file, 'wb') as f:
                pickle.dump(total_trees, f)
                    
            done_task_id.append(task.task_id) 
            pbar.update(1)
        pbar.close()

    n_traj = len(total_examples)
    n_success = sum([ 1. if traj['reward']==1.0 else 0. for traj in total_examples])
    rewards = [traj['reward'] for traj in total_examples]
    # i = 0
    # best_rewards = []
    # for r in rewards:
    #     candidiates = []
    #     if i<3:
    #         candidiates.append(r)
    #     else:
    #         best_rewards.append(max(candidiates))
    #         candidiates = []
    #         i = i%3
            
        # i += 1
    print(f"Finally, The Number of Successful Trajectories: {n_success} / {n_traj} ")
    print(f"Average Reward: {sum(rewards) / n_traj}")
    #print("avg best reward:", sum(best_rewards) / len(best_rewards))
    wandb.log({"n_traj": n_traj, "n_success": n_success, "avg_reward": sum(rewards) / n_traj})
    # save all trajs
    if args.debug:
        traj_file = f'{root_dir}/data/train/explore/debug2.jsonl'
        tree_file = f'{root_dir}/data/train/explore/debug2.pkl'
        
    with open(traj_file, 'w') as f:
        #print("total_examples:",total_examples)
        json.dump(total_examples, f)
            
    with open(tree_file, 'wb') as f:
        pickle.dump(total_trees, f)
        
if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    #parser.add_argument("--max_num_examples", type=int, default=None, help="maximum number of examples to evaluate.")
    parser.add_argument("--slice_id", type=int, default=0)
    parser.add_argument("--slice_num", type=int, default=1, help="Evaluation part.")
    parser.add_argument("--save_dir", type=str)
    #parser.add_argument("--model_name_or_path", type=str, default=None, help="if specified, we will load the model to generate the predictions.")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="batch size for evaluation.")
    parser.add_argument("--best_of_N", type=int, default=10, help="Number of best actions to select.")
    parser.add_argument("--n_trajs", type=int, default=1, help="number of workers for evaluation.")
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
        "--samples_per_depth",
        type=int,
        default=1,
        help="Num of explored tree nodes at each depth.",
    )

    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
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
        "--num_icl_examples",
        type=int,
        default=1,
        help="Number of ICL examples to generate.",
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth of the tree.",
    )
    parser.add_argument(
        "--qnet_path",
        type=str,
        default=None,
        help="Path to the QNet model.",
    )
    parser.add_argument(
        "--sample_mode",
        type=str,
        default='epsilon_greedy',
        help="Sampling mode for the trajectories.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='data/train/explore/sft',
        help="Output directory for the trajectories and trees.",
    )
    args = parser.parse_args()
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)