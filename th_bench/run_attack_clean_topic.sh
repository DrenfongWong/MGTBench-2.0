attack_type="clean"

# 定义所有可能的detectLLM选项
detectLLM_options=("Moonshot" "gpt35" "Mixtral" "Llama3" "gpt-4omini")
for model in "${detectLLM_options[@]}"; do
	echo "Executing attack with model $model"
	CUDA_VISIBLE_DEVICES=1,6 python attack_run_topic.py --detectLLM $model
	echo "Finished execution with model $model"
	echo "--------------------------------"
done
