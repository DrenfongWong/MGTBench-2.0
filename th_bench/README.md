#Text_Humanization_Benchmark

## Quick Start

### clean data preparetion
```bash
# create env for clean data generation $attack in ("dipper" "recursive_dipper" "token_ensemble" "raft")
conda create -n attack_run python=3.9
conda activate attack_run
pip install -r requirements_for_attack_run.txt
# generate clean mixed human and $model data file for MGTbench2 in cat, $model in ("Moonshot" "gpt35" "Mixtral" "Llama3" "gpt-4omini")
python attack_run.py --detectLLM $model --datatype mgt2
# generate clean mixed human and $model data file for MGTbench2 in topic, $model in ("Moonshot" "gpt35" "Mixtral" "Llama3" "gpt-4omini")
python attack_run.py --detectLLM $model --datatype mgt2_topic
# please follow https://github.com/xinleihe/MGTBench to download the orginal data file, you can download them from [Google Drive](https://drive.google.com/drive/folders/1p4iBeM4r-sUKe8TnS4DcYlxvQagcmola?usp=sharing), then put them into ./data dir
# generate clean mixed human and $model data file for MGTbench, $model in ("ChatGPT-turbo" "ChatGLM" "Dolly" "ChatGPT" "GPT4All" "Claude" "StableLM")
python attack_run.py --detectLLM $model --datatype mgt1
```

### attack in ("dipper" "recursive_dipper" "token_ensemble" "raft"), the attack file is pre_dataset/clean_${model}_${attack}.csv
```bash
# $attack in ("dipper" "recursive_dipper" "token_ensemble" "raft")
conda activate attack_run
python attack_run.py --detectLLM $model --attack $attack --datatype mgt1/mgt2/mgt2_topic
# if you want to collect time and gpu usage in attack, use this cmd
python attack_run_topic_gpu.py --detectLLM $model --attack $attack --datatype mgt1/mgt2/mgt2_topic
``` 

### hmgc attack, the attack file is pre_dataset/clean_${model}_${dataset}.csv 
```bash
conda env create -n hmgc python=3.9
conda activate hmgc
pip install -r requirements_for_hmgc.txt
python flint_attack.py  --model_name_or_path ${model_dir} --output_dir ${output_dir} --attacking_method dualir --dataset ${model}_${dataset}
# if you want to collect time and gpu usage in attack, use this cmd
python flint_attack_topic_gpu.py  --model_name_or_path ${model_dir} --output_dir ${output_dir} --attacking_method dualir --dataset ${model}_${dataset}
```

### benchmark using MGTBENCH2.0, follow the ../README to do the benchmark
```bash
cd ../
conda env create -f mgtbench2.yml;
conda activate mgtbench2;
# you may need mirror for faster installation
pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
pip install -r requirements.txt
cd run
python benchmark.py --csv_path ${the_result_file_path} --method ll --detect_LLM ${LLM} --localdata ${attacked_csv_flie_path}
# calculate the quality of attack
python text_quality_cal.py --detectLLM Moonshot --dataset Physics --attack raft
```
