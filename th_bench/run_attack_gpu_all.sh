attack_type="clean"

# 定义所有可能的detectLLM选项
detectLLM_options=("Moonshot" "gpt35" "Mixtral" "Llama3" "gpt-4omini")
attack_options=("token_ensemble" "dipper" "recursive_dipper" "raft")
for attack in "${attack_options[@]}"; do
	for model in "${detectLLM_options[@]}"; do
		echo "Executing attack with model $model"
		CUDA_VISIBLE_DEVICES=2,4 python attack_run_topic_gpu.py --detectLLM $model --attack $attack
		echo "Finished execution with model $model"
		echo "--------------------------------"
	done
done
