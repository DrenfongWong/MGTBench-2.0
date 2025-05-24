attack_type="clean"

# 定义所有可能的detectLLM选项
detectLLM_options=("gpt35" "Mixtral" "Llama3" "Moonshot" "gpt35" "gpt-4omini")
datasets=("Physics" "Medicine" "Biology" "Electrical_engineering" "Computer_science" "Literature" "History" "Education" "Art" "Law" "Management" "Philosophy" "Economy" "Math" "Statistics" "Chemistry")
for model in "${detectLLM_options[@]}"; do
	for dataset in "${datasets[@]}"; do
		echo "Executing attack with model $model $dataset"
		base_dir="/data1/lyl/mgtout/roberta-base/LM-D_${model}_${dataset}"
		first_subdir=$(ls -d "$base_dir"/*/ | head -n 1)
		echo "base_dir ${base_dir} first_subdir ${first_subdir}"
		CUDA_VISIBLE_DEVICES=1,6 python flint_attack.py  --model_name_or_path ${first_subdir} --output_dir flint_result --attacking_method dualir --dataset ${model}_${dataset} 
		echo "Finished execution with model $model $dataset"
	done
	echo "--------------------------------"
done
