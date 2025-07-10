
import json
import os

data_dir = './data/train/alfworld/'
model_name = 'qlass-Llama-2-7b-chat-hf-alfworld-sft'
inference_type = 'q_without_perturb'
eval_name = 'debug_bon2_dev'
slice_num = 4
file_list = []
for i in range(slice_num):
    file_path = os.path.join(data_dir, model_name, inference_type, eval_name, f"{i}of{slice_num}_slices_bon_traj.jsonl")
    file_list.append(file_path)

total_max = 0
total_max_list = 0
total_bon = 0
total_bon_list = 0
total_first = 0
total_first_list = 0
for file in file_list:
    test_data = json.load(open(file, "r"))
    print(len(test_data))
    gap = 3
    print("gap:", gap)
    reward_list = [traj['reward'] for traj in test_data]
    print("avg reward:", sum(reward_list) / len(reward_list))
    mult_success_list = []
    first_success_list = []
    bon_success_list = []
    for i in range(0, len(reward_list), gap):
        msn = 0
        if reward_list[i] == 1.0:
            first_success_list.append(1.0)
        else:
            first_success_list.append(0.0)

        for j in range(i, i + gap):
            if reward_list[j] == 1.0:
                msn += 1
        if msn > 1:
            mult_success_list.append(1.0)
        else:
            mult_success_list.append(0.0)
        
        if msn > 0:
            bon_success_list.append(1.0)
        else:
            bon_success_list.append(0.0)

    # Compute the average of the max values
    avg_max_reward = sum(mult_success_list) / len(mult_success_list)
    total_max += sum(mult_success_list)
    total_max_list += len(mult_success_list)
    total_first += sum(first_success_list)
    total_first_list += len(first_success_list)
    total_bon += sum(bon_success_list)
    total_bon_list += len(bon_success_list)
    print(f"avg max of every {gap} rewards:", avg_max_reward)

print("avg total max:", total_max / total_max_list)
print("avg total first:", total_first / total_first_list)
print("avg total bon:", total_bon / total_bon_list)
