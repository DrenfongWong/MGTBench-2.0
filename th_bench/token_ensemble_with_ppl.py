from transformers import AutoModelForCausalLM, AutoTokenizer, BertModel, BertTokenizer
from nltk.tokenize import word_tokenize
import tqdm
import os
import string
import time
import random
import torch
import json
import numpy as np
from scipy.spatial.distance import cosine

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calculate_ppl(model, tokenizer, text):
    # Calculate text perplexity using the transformers library pipeline
    ppl_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
    ppl = ppl_pipeline.compute_perplexity(text)
    return ppl

def calculate_bert_similarity(model, tokenizer, text1, text2):
    # Calculate semantic similarity between two texts using BERT
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)
    
    embeddings1 = outputs1.last_hidden_state[:, 0, :].detach().numpy()
    embeddings2 = outputs2.last_hidden_state[:, 0, :].detach().numpy()
    
    similarity = 1 - cosine(embeddings1, embeddings2)
    return similarity

def completion_judgement(tokenizer, generated_text, reference_text, bert_model, bert_tokenizer):
    # Check if the text contains the end token of the pre-trained model
    if tokenizer.eos_token in generated_text:
        return True
    # Calculate similarity between generated text and reference text
    similarity = calculate_bert_similarity(bert_model, bert_tokenizer, generated_text, reference_text)
    if similarity > 0.8:  # Assuming similarity threshold is 0.8
        return True
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

    # Load BERT model and tokenizer for similarity calculation
    bert_model_name = 'bert-base-uncased'
    bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name, cache_dir=cache_dir)
    bert_model = BertModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
    bert_model.to(cuda_name[0])

    for idx, dd in tqdm.tqdm(enumerate(data['text']), total=len(data['text'])):
        if data['label'][idx] == 0:
            continue
        original = data['text'][idx]
        token_list = word_tokenize(original)
        tokens = token_list[:30]
        incomplete = " ".join(tokens)
        flag_completion = False
        
        prompt = f'{incomplete}'
        reference_text = original  # Assume the reference text is the original text
        
        T1 = time.time()
        times = 0
        while not flag_completion:
            T3 = time.time()
            
            next_tokens = []
            ppls = []
            for model_info in model_info_list:
                model_name, tokenizer, model, idn = model_info
                
                inputs = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(cuda_name[idn])
                
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, num_return_sequences=1, min_length=1, do_sample=True,
                                         pad_token_id=tokenizer.eos_token_id,
                                         eos_token_id=tokenizer.eos_token_id,
                                         return_dict_in_generate=True, output_scores=True)
                
                input_length = 1 if model.config.is_encoder_decoder else inputs.input_ids.shape[1]
                generated_tokens = outputs.sequences[:, input_length:]
                temp_output = tokenizer.decode(generated_tokens[0])
                next_token = temp_output.strip()
                
                next_tokens.append(next_token)
                ppls.append(calculate_ppl(model, tokenizer, prompt + " " + next_token))
            min_ppl_idx = np.argmin(ppls)
            next_token = next_tokens[min_ppl_idx]
            
            prompt = prompt + " " + next_token
            
            flag_completion = completion_judgement(tokenizer, prompt, reference_text, bert_model, bert_tokenizer)
            
            if flag_completion:
                break
            
            times += 1
            if times > 4:
                flag_completion = True
            
            T4 = time.time()
            time_span = np.round((T4 - T3),2)
            print("\n", idx, prompt,"\n", next_token,"\n", flag_completion ,"\n")

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
def run_token_ensemble_with_ppl(data):
    model_list = ['gpt2-xl', 'facebook/opt-2.7b', 'EleutherAI/gpt-neo-2.7B', 'EleutherAI/gpt-j-6B']
    cuda_name = ['cuda:0', 'cuda:1', 'cuda:0', 'cuda:1']
    return complete_text_with_model(data, model_list, cuda_name)
