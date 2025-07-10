# import json
# import time
# import logging
# from typing import List, Dict, Union, Any
# import requests
# # from fastchat.model.model_adapter import get_conversation_template
# import os
# import sys
# sys.path.append("/home/yaotang/agent/fastchat")
# from requests.exceptions import Timeout, ConnectionError
# from eval.utils import generate_completions, load_hf_lm_and_tokenizer
# from .base import LMAgent

# logger = logging.getLogger("agent_frame")


# def _add_to_set(s, new_stop):
#     if not s:
#         return
#     if isinstance(s, str):
#         new_stop.add(s)
#     else:
#         new_stop.update(s)


# class LlamaAgent(LMAgent):
#     """This agent is a test agent, which does nothing. (return empty string for each action)"""

#     def __init__(
#         self,
#         config
#     ) -> None:
#         super().__init__(config)
#         self.controller_address = config["controller_address"]
#         self.model_name = config["model_name"]
#         print("Loading model and tokenizer from",config["model_name"],"...")
#         model, tokenizer = load_hf_lm_and_tokenizer(
#             model_name_or_path= config["model_name"], 
#             tokenizer_name_or_path= config["model_name"], 
#             load_in_8bit=False, 
#             load_in_half=True,
#         )
#         self.model = model
#         self.tokenizer = tokenizer
#         self.temperature = config.get("temperature", 0)
#         self.batch_size = config.get("batch_size", 1)
#         print("temperature:",self.temperature)
#         self.max_new_tokens = config.get("max_new_tokens", 512)
#         self.top_p = config.get("top_p", 0)

#     def __call__(self, messages: List[dict]) -> str:
#         # raise NotImplementedError
    
#         # controller_addr = self.controller_address
#         # worker_addr = controller_addr
#         # if worker_addr == "":
#         #     raise ValueError
#         gen_params = {
#             "model": self.model_name,
#             "temperature": self.temperature,
#             "max_new_tokens": self.max_new_tokens,
#             "echo": False,
#             "top_p": self.top_p,
#         }
        
        
#         # print("messages:",messages)
#         def convert_messages_to_prompts(messages):
#             message_text = ""
#             for message in messages:
#                 # if message["role"] == "system":
#                 #     message_text += "<|system|>\n" + message["content"].strip() + "\n"
#                 if message["role"] == "user":
#                     message_text += "<|user|>\n" + message["content"].strip() + "\n"
#                 elif message["role"] == "assistant":
#                     message_text += "<|assistant|>\n" + message["content"].strip() + self.tokenizer.eos_token + "\n"
#                 else:
#                     raise ValueError(f"Unknown role: {message['role']}")
#                 return message_text
        
#         prompts = [convert_messages_to_prompts(messages)]
#         print("prompts:",prompts)
#         # for i in tqdm(range(16)):
#         new_line_token = self.tokenizer.encode("\n", add_special_tokens=False)[-1]
#         stop_id_sequences = [[new_line_token]]
#         #     #  generations = generate_completions
#         #     #  (model=self.model, tokenizer=self.tokenizer, prompts=messages, gen_params=gen_params)
#         #     # import pdb; pdb.set_trace(header="Before generate_completions")
#         #     # sample_probability = 0.5
#         #     # do_sample = random.random() < sample_probability
#         # generation_args = {
#         #         'temperature': self.temperature, 
#         #         "top_p": self.top_p,
#         #         # 'do_sample': False,
#         #     }
#         generation_args = {
#                 'temperature': 1.0,  # 2.0
#                 'do_sample': False,
#             }
#         completions = generate_completions(
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 prompts=prompts,
#                 max_new_tokens=self.max_new_tokens,
#                 batch_size=self.batch_size,
#                 stop_id_sequences=stop_id_sequences,
#                 **generation_args,
#             )
#         print("completions:",completions)
#         return completions[0].strip()
#         # conv = get_conversation_template(self.model_name)
#         # for history_item in messages:
#         #     role = history_item["role"]
#         #     content = history_item["content"]
#         #     if role == "user":
#         #         conv.append_message(conv.roles[0], content)
#         #     elif role == "assistant":
#         #         conv.append_message(conv.roles[1], content)
#         #     else:
#         #         raise ValueError(f"Unknown role: {role}")
#         # conv.append_message(conv.roles[1], None)
#         # prompt = conv.get_prompt()
#         # new_stop = set()
#         # _add_to_set(self.stop_words, new_stop)
#         # _add_to_set(conv.stop_str, new_stop)
#         # gen_params.update(
#         #     {
#         #         "prompt": prompt,
#         #         "stop": list(new_stop),
#         #         "stop_token_ids": conv.stop_token_ids,
#         #     }
#         # )
#         # headers = {"User-Agent": "FastChat Client"}
#         # for _ in range(3):
#         #     try:
#         #         response = requests.post(
#         #             controller_addr + "/worker_generate_stream",
#         #             headers=headers,
#         #             json=gen_params,
#         #             stream=True,
#         #             timeout=120,
#         #         )
#         #         text = ""
#         #         for line in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
#         #             if line:
#         #                 data = json.loads(line)
#         #                 if data["error_code"] != 0:
#         #                     assert False, data["text"]
#         #                 text = data["text"]
#         #         return text
#         #     # if timeout or connection error, retry
#         #     except Timeout:
#         #         print("Timeout, retrying...")
#         #     except ConnectionError:
#         #         print("Connection error, retrying...")
#         #     time.sleep(5)
#         # else:
#         #     raise Exception("Timeout after 3 retries.")
        
# # def lumos_iterative_singlemodel_gentraj(messages, gt_ans, depth, model, tokenizer, args):
# #     random.seed(42)

# #     new_line_token = tokenizer.encode("\n", add_special_tokens=False)[-1]

# #     subgoals, actions, all_subgoals_actions, subgoal_results, solved_idx = [], [], [], [], []
# #     prompts = []
    
# #     # assert it is ended with user
# #     new_trajs = messages[:depth]
# #     assert new_trajs[-1]["role"] == "user"
# #     final_reward = 0
    
# #     # init prompts using test_data
# #     prompts = [_concat_messages(new_trajs, tokenizer).strip()] # only one sample
# #     prompts[0] += '\n<|assistant|>\n'
    
# #     # compute current index of subgoal
# #     cur_subgoal_idx, cur_ans = _extract_result(new_trajs[-1]["content"])
# #     if cur_subgoal_idx is None:
# #         cur_subgoal_idx = 0
        
# #     # init the results
# #     all_subgoals_actions = extract_and_execute_actions(new_trajs)
# #     all_subgoals_actions[0]["answer"] = gt_ans
# #     print(all_subgoals_actions)
# #     # import pdb; pdb.set_trace()
    
# #     for i in tqdm(range(16)):
# #         stop_id_sequences = [[new_line_token]]
# #         if solved_idx:
# #             continue
# #         if i % 2 == 0:  # plan
# #             print('#'*20)
# #             print(f"Begin to plan, i={i}")
# #             print('#'*20)

# #             if i != 0:
# #                 # only report the last execution result for each turn
# #                 feedback = f"The executed result for Subgoal {int(cur_subgoal_idx)} is {all_subgoals_actions[0]['actions'][-1].split('=')[-1] if all_subgoals_actions[0]['actions'] else 'None'}. Should we stop planning?\n"

# #                 prompts[0] += "\n\n<|user|>\n" + feedback + "<|assistant|>\n"
# #                 new_trajs.append({'role': 'user', 'content': feedback.strip()})
                
# #         else:  # ground
# #             print('#'*20)
# #             print(f"Begin to ground, i={i}")
# #             print('#'*20)
# #             grounding_info = ""
# #             if i == 1:
# #                 grounding_info += "<|user|>\nSubgoal to be grounded: " + subgoals[0].strip() + "\n\n"
# #             else:
# #                 grounding_info += f"\n\n<|user|>\nSubgoal to be grounded: {subgoals[0].split('No, I will keep planning.')[-1].strip()}\n<|assistant|>\n"
# #             prompts[0] += grounding_info    
# #             grounding_info = grounding_info.replace("<|user|>\n", "").replace("<|assistant|>\n", "")
# #             new_trajs.append({'role': 'user', 'content': grounding_info.strip()})
        
        
# #         # import pdb; pdb.set_trace(header="Before generate_completions")
# #         sample_probability = 0.5
# #         do_sample = random.random() < sample_probability


# #         if i % 2 == 0:
# #             generation_args = {
# #                 'temperature': 1.0,  # 2.0
# #                 'do_sample': False,
# #             }
# #             # generation_args = {
# #             #     'temperature': 1.0,
# #             #     'num_beams': 5,
# #             #     'num_return_sequences': 3,
# #             # }
# #             completions = generate_completions(
# #                 model=model,
# #                 tokenizer=tokenizer,
# #                 prompts=prompts,
# #                 max_new_tokens=512,
# #                 batch_size=args.eval_batch_size,
# #                 stop_id_sequences=stop_id_sequences,
# #                 **generation_args,
# #             )
# #             print(completions)
# #         else:
# #             completions = generate_completions(
# #                 model=model,
# #                 tokenizer=tokenizer,
# #                 prompts=prompts,
# #                 max_new_tokens=512,
# #                 batch_size=args.eval_batch_size,
# #                 stop_id_sequences=stop_id_sequences,
# #                 **generation_args,
# #             )
# #             print(completions)
        
# #         # import pdb; pdb.set_trace(header="After generate_completions")
        
# #         if i % 2 == 0:  # plan
# #             subgoals = completions
# #             if solved_idx:
# #                 continue
# #             prompts[0] += subgoals[0].strip() # 0423
# #             new_trajs.append({'role': 'assistant', 'content': subgoals[0].strip()})
# #             if "Yes, I will stop planning" in subgoals[0].strip():
# #                 solved_idx.append(0)
# #         else:            
# #             if solved_idx:
# #                 continue
# #             actions = completions
# #             prompts[0] += actions[0].strip() # 0424, add actions to prompts
# #             new_trajs.append({'role': 'assistant', 'content': actions[0].strip()})
            
# #             all_subgoals_actions[0]["subgoals"].append(subgoals[0].strip())
# #             cur_subgoal_idx = int(subgoals[0].strip().split(':')[0].split("No, I will keep planning.")[-1].strip().split(" ")[-1]) # slightly different from the eval one, this one is the number
            
# #             for k, action in enumerate(actions[0].strip().split('; ')): # the execution results are added to the user's message
# #                 try:
# #                     action = action.split("No, I will stop planning.")[0]
# #                     results_variable, execution_results = execute(action, all_subgoals_actions[0]["results"])
# #                     assert re.match(r'^R\d+$', results_variable) # 0503
                    
# #                     all_subgoals_actions[0]["results"][results_variable] = execution_results
# #                     # print(f'execution_results: {execution_results}')
# #                 except Exception as e:
# #                     traceback.print_exc()
# #                     execution_results = "None"
# #                     pass
                
# #                 try:
# #                     if all_subgoals_actions[0]["results"]:
# #                         if isinstance(execution_results, str):
# #                             all_subgoals_actions[0]["actions"].append(action.strip() + " = " + execution_results)
# #                         else:
# #                             all_subgoals_actions[0]["actions"].append(action.strip() + " = " + ", ".join(execution_results))
# #                     else:
# #                         all_subgoals_actions[0]["actions"].append(action.strip())
# #                 except:
# #                     all_subgoals_actions[0]["actions"].append(action.strip())

# #     corr, llm_acc, final_reward = 0, 0, 0
# #     with open(os.path.join(args.save_dir, "predictions_iterative.jsonl"), "w") as f:
# #         for subgoal_action in all_subgoals_actions:
# #             if subgoal_action["results"]:
# #                 final_variable = 'R' + str(max(int(k[1:]) for k in subgoal_action["results"].keys()))
# #                 print(subgoal_action["results"][final_variable].strip())
# #                 print(subgoal_action["answer"])
# #                 print(subgoal_action["results"][final_variable].strip() == subgoal_action["answer"])
# #                 # import pdb; pdb.set_trace()
# #                 if subgoal_action["results"][final_variable].strip() == subgoal_action["answer"]:
# #                     corr += 1
# #                     final_reward = 1
# #                 # llm_acc += llm_accuracy_score(subgoal_action["question"], subgoal_action["results"][final_variable].strip(), subgoal_action["answer"])
# #                 f.write(json.dumps({"question": subgoal_action["question"], "inter_pred":subgoal_action["results"],"pred": subgoal_action["results"][final_variable], "answer": subgoal_action["answer"], "subgoals": subgoal_action["subgoals"], "actions": subgoal_action["actions"]})+'\n')
# #                 # if llm_acc == 1:
# #                 #     final_reward = 1
                    

# #     print("Acc:", corr)
# #     if not all_subgoals_actions[0]["results"]:
# #         return new_trajs, 0, "No Answer", all_subgoals_actions[0]["answer"]
# #     else:
# #         return new_trajs, final_reward, all_subgoals_actions[0]["results"][final_variable], all_subgoals_actions[0]["answer"]

