model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=4  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4
accumulation_step=$((${batch_size}/${node_num}/${micro_batch_size}))

sft_model_name=${exp_name}-${model_name}-${task}-sft
sg_worker_port=21001

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs/${exp_name}-sg_worker.log 2>&1 &

sg_worker_pid=$!
sleep 60

python -m qlass.inference --agent_config sglang_sft --model_name ${sft_model_name} --exp_config ${task} --split dev --num_icl_examples 1 --exp_name qlass_eval_sft --force_first