attack_type="clean"
n=0
detectLLM_options=("Mixtral" "Llama3" "Moonshot" "gpt35" "gpt-4omini")
datasets=("Humanities" "Social_sciences" "STEM")
for model in "${detectLLM_options[@]}"; do
	for dataset in "${datasets[@]}"; do
		for length in {100..1000..100}; do
			((n=n+1))
			if [ $n -lt 1 ]; then
				echo "Skipping execution with model $model and attack type $attack (n=$n)"
				continue
			fi
			echo "Executing attack with model $model $dataset $length"
			CUDA_VISIBLE_DEVICES=0 python run_prompt.py  --file pre_dataset/clean_${length}_${model}_${dataset}.csv --model /data1/models/Llama-2-7b-chat-hf
			echo "Finished execution with model $model $dataset $length"
		done
	done
	echo "--------------------------------"
done
