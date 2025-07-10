model_name=Llama-2-7b-chat-hf
task=alfworld # webshop, scienceworld, alfworld
exp_name=qlass

node_num=4  # number of GPUs

model_path=${MODEL_PATH} # path to the original LLM
save_dir=${MODEL_PATH}    # checkpoint save path


sft_model_name=${exp_name}-${model_name}-${task}-sft

q_model_name=${exp_name}-${model_name}-${task}-Q

sg_worker_port=21001
sg_worker_port2=21002
sg_worker_port3=21003
sg_worker_port4=21004

CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port} >> logs/${exp_name}-sg_worker_explore.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port2} >> logs/${exp_name}-sg_worker_explore2.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port3} >> logs/${exp_name}-sg_worker_explore3.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server --model-path ${save_dir}${sft_model_name} --port ${sg_worker_port4} >> logs/${exp_name}-sg_worker_explore4.log 2>&1 &

sleep 30

BON=2
ICL=1
SPLT=dev
data_prefix=debug

slice_num=4
for (( slice_id=0; slice_id<slice_num; slice_id++ )); do
  CUDA_VISIBLE_DEVICES=$((slice_id+1)) python qlass/q_guided_inference.py \
    --agent_config sglang \
    --agent_path qlass/configs/model/ \
    --qnet_path "${save_dir}${q_model_name}" \
    --exp_name "${exp_name}" \
    --exp_path qlass/configs/task/ \
    --exp_config "${task}" \
    --split "${SPLT}" \
    --slice_num "${slice_num}" \
    --slice_id "${slice_id}" \
    --model_name "${explore_model_name}" \
    --num_icl_examples "${ICL}" \
    --sample_mode bon \
    --best_of_N "${BON}" \
    --n_trajs 3 \
    --force_first \
    --disable_perturb \
    --output_dir "data/train/${task}/${explore_model_name}/q_without_perturb/${data_prefix}_bon${BON}_${SPLT}/" &

  sleep 10
done