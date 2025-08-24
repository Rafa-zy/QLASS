# QLASS
### 🛠️ Set up environment
```
conda create -n qlass_dev python=3.10
conda activate qlass_dev

pip install -r requirements.txt
./setup.sh
pip install flash-attn==2.5.8 --no-build-isolation 
pip install sentencepiece 
pip install alfworld==0.3.5 
pip install cleantext 
pip install openai==0.28.1 
pip install gym 
pip install selenium 
pip install omegaconf 
pip install protobuf
pip install termcolor colorama 
pip install ipdb
pip install rank_bm25          
pip install matplotlib
pip install pyserini 
pip install scienceworld   
cd envs/webshop
python setup.py install
pip install uv
uv pip install "sglang[all]>=0.4.8"
apt-get install -y libgl1-mesa-glx
```

### 📑 Data Setup
```
### download sft json
huggingface-cli download qlass/qlass_sft_data

### download alfworld data
cd eval_agent/data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip
```

After setup, the structure of the data folder should look like
```
 
 ├── data/train/
 │   ├── webshop
 │   │   ├── explore                 # Used to store 1-self-explorated output
 │   │   ├── guided_explore          # Used to store 2-1 Q guided exploration output
 │   │   └── webshop_sft.json        # JSON file containing fine-tuning data environment.
 │   ├── alfworld
 │   │   ├── explore
 │   │   ├── guided_explore
 │   │   └── alfworld_sft.json       
 │   └── sciworld
 │       ├── explore
 │       ├── guided_explore
 │       └── sciworld_sft.json       

```
### ⚙️ Resource Requirements
Our scripts are suitable for `4*A6000/A100/H100/A800/H800`. If you want to run on one or two gpus, you can change the logic in the scripts.

### 🚀 Run the Q-guided inference
We show how to directly use the well-trained QNet to run inference.
First Download SFT model from https://huggingface.co/qlass/qlass-Llama-2-7b-chat-hf-alfworld-sft and Q-Net from https://huggingface.co/qlass/qlass-Llama-2-7b-chat-hf-alfworld-Q and put them in `MODEL_PATH`
Before you start, make sure you have correct `sglang*.json` in `configs/agent/model`
```
bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld, this step will generate eval results files in {output_dir}

python ./qlass/calc_results.py ## collect the final results (you can change the path according to {output_dir} inside the code)
```

### 🎮 Run the whole pipeline
Download Llama-2-7b-chat-hf from https://huggingface.co/meta-llama/Llama-2-7b-chat-hf and put it in `MODEL_PATH`

Before you start, make sure you have correct `sglang*.json` in `configs/agent/model`

```
### SFT the model
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/sft_7b_alfworld.sh

### Eval sft model (you can change split to test on either dev/test set)
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/eval_sft_7b_alfworld.sh

### Exploration
MODEL_PATH=/path/to/your/model bash ./qlass/scripts/explore_7b_alfworld.sh

### Collect Q Data
bash ./qlass/scripts/collect_q.sh

### Train Q-Net
bash ./qlass/scripts/train_qnet_7b.sh

### Q-guided inference
bash ./qlass/scripts/eval_q_wo_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld, this step will generate eval results files in {output_dir}

python ./qlass/calc_results.py ## collect the final results (you can change the path according to {output_dir} inside the code)

### if you want to use q-inference with perturbation
export OPENAI_ORG={YOUR_ORG}
export OPENAI_API_KEY={YOUR_KEY}
bash ./qlass/scripts/eval_q_perturb_7b_alfworld.sh ## we use no perturbation version for alfworld


```

### 🔧 Some Common Issues & solutions
```
BUG: libstdc++.so.6: version `GLIBCXX_3.4.29' not found
SOLUTION: https://github.com/pybind/pybind11/discussions/3453

BUG: Exception: Unable to find javac
SOLUTION: https://stackoverflow.com/questions/5736641/ant-unable-to-find-javac-java-home-wont-set-on-ubuntu/37201765#37201765

```

### 🌹 Acknowledgement
We borrowed some implementations from https://github.com/Yifan-Song793/ETO and https://github.com/sgl-project/sglang. Thanks for their great work!

### 📖 Citation

If you find this repo helpful, please cite out paper:

```
@article{lin2025qlass,
  title={QLASS: Boosting Language Agent Inference via Q-Guided Stepwise Search},
  author={Lin, Zongyu and Tang, Yao and Yao, Xingcheng and Yin, Da and Hu, Ziniu and Sun, Yizhou and Chang, Kai-Wei},
  journal={arXiv preprint arXiv:2502.02584},
  year={2025}
}
```