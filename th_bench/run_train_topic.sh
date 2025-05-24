attack_type="clean"

# 定义所有可能的detectLLM选项
detectLLM_options=("gpt35" "Moonshot" "Mixtral" "Llama3" "gpt-4omini")
for model in "${detectLLM_options[@]}"; do
	echo "Executing attack with model $model"
	CUDA_VISIBLE_DEVICES=3,4 python train_model_for_topic.py --detectLLM $model
	echo "Finished execution with model $model"
	echo "--------------------------------"
done
