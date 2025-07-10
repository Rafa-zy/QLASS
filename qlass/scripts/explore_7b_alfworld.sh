model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

num_workers=16   # number of inference workers
num_server=4

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path

# Part 1: SFT stage
sft_data_path="data/train/${task}/${task}_sft.json"
batch_size=64
micro_batch_size=4

sft_model_name=${exp_name}-${model_name}-${task}-sft
sg_worker_port=21001
sg_worker_port2=21005
sg_worker_port3=21003
sg_worker_port4=21004

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs/${exp_name}-sg_worker_explore1.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port2} >> logs/${exp_name}-sg_worker_explore2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port3} >> logs/${exp_name}-sg_worker_explore3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port4} >> logs/${exp_name}-sg_worker_explore4.log 2>&1 &

sleep 60

for ((i=0;i<${num_workers};i=i+1)); do
    echo "Start worker ${i} querying server $((i%$num_server))"
    python qlass/explore_sft_agent.py \
        --agent_config sglang_explore$((i%$num_server)) \
        --agent_path qlass/configs/model/ \
        --exp_name ${exp_name} \
        --exp_path qlass/configs/task/ \
        --exp_config ${task} \
        --split train \
        --slice_num ${num_workers} \
        --slice_id ${i} \
        --model_name ${sft_model_name} \
        --max_depth 8 \
        --min_prune_depth 3 \
        --num_icl_examples 0 \
        --samples_per_depth 2 \
        --output_dir data/train/${task}/explore_7b_sft_d8_i0_s2_mpr3/ &
done
