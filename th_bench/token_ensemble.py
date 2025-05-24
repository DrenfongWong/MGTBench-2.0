from transformers import AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import word_tokenize
import tqdm
import os
import string
import time
import random
import torch
import json
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
                    # 如果您使用的是torch的CUDA设备，还需要设置以下内容
    torch.cuda.manual_seed_all(seed)

def select_model_random_loaded(model_info_list):
    model_info = random.choice(model_info_list)
    model_name = model_info[0]
    tokenizer = model_info[1]
    model = model_info[2]
    idx = model_info[3]
    return model_name, tokenizer, model, idx

def completion_judgement(incomplete, next_token):
    tokens = word_tokenize(incomplete)
    if (len(tokens) >= 400 and next_token == '.') or len(tokens) > 500:
        return True
    else:
        return False

def complete_text_with_model(data, model_list, cuda_name, max_new_tokens=1):
    set_seed(42)
#    cache_dir = '/data1/zjy/Text_Humanization_Benchmark'
    model_info_list = []
    
    # Load models, tokenizer and basic EOS and PAD token_id in the GPU first
    for i, model_name in enumerate(model_list):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.to(cuda_name[i])
        model_info_list.append([model_name, tokenizer, model, i])

    for idx, dd in tqdm.tqdm(enumerate(data['text']), total=len(data['text'])):
        if data['label'][idx] == 0:
            continue
        original = data['text'][idx]
        token_list = word_tokenize(original)
        tokens = token_list[:30]
        incomplete = " ".join(tokens)
        flag_completion = False
        
        prompt = f'{incomplete}'
        
        T1 = time.time()
        times = 0
        while not flag_completion:
            T3 = time.time()
            
            model_name, tokenizer, model, idn = select_model_random_loaded(model_info_list)

            inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(cuda_name[idn])
            
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, min_length=1, do_sample=True,
                                     pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     return_dict_in_generate=True, output_scores=True)
            
            input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
            generated_tokens = outputs.sequences[:, input_length:]
            temp_output = tokenizer.decode(generated_tokens[0])
            cut_token_number = 0 
            temp_token_list = word_tokenize(temp_output) 
            temp_tokens = temp_token_list[cut_token_number:]
            cut_answer = " ".join(temp_tokens)
            flag_completion = completion_judgement(prompt, cut_answer)
            if cut_answer == "":
                times += 1;
                if times > 4:
                    flag_completion = True
            print("\n", idx, prompt,"\n", cut_answer,"\n", flag_completion ,"\n")
            if cut_answer in list(string.punctuation):
                prompt = prompt + cut_answer
            else: 
                prompt = " ".join([prompt, cut_answer])

            T4 = time.time()
            time_span = np.round((T4 - T3),2)

        data['text'][idx] = prompt

        T2 = time.time()
        time_span = np.round((T2 - T1),2)
        
        print('---------------------------------------------')
        print('The running time is', time_span, ' seconds.')
        print('---------------------------------------------')

    # Save the results as JSON
    if not os.path.exists("results"):
        os.mkdir("results")
    times = time.time()
    with open(f'results/token_ensemble_{times}.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    torch.cuda.empty_cache()
    return data

# Example usage
def run_token_ensemble(data):
    model_list = ['gpt2-xl', 'facebook/opt-2.7b', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B']
    cuda_name = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']
    return complete_text_with_model(data, model_list, cuda_name)
