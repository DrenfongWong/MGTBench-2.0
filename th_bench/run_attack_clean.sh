attack_type="clean"

detectLLM_options=("Moonshot" "gpt35" "Mixtral" "Llama3" "gpt-4omini")
detectLLM_options_mgt=("ChatGPT-turbo" "ChatGLM" "Dolly" "ChatGPT" "GPT4All" "StableLM" "Claude")
for model in "${detectLLM_options[@]}"; do
	echo "Executing attack with model $model"
	CUDA_VISIBLE_DEVICES=1 python attack_run.py --detectLLM $model
	echo "Finished execution with model $model"
	echo "--------------------------------"
done

for model in "${detectLLM_options_mgt[@]}"; do
	echo "Executing attack with model $model"
	CUDA_VISIBLE_DEVICES=0  python attack_run.py --detectLLM $model --datatype mgt1
	echo "Finished execution with model $model"
	echo "--------------------------------"
done
