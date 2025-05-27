from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import csv
from rouge_score import rouge_scorer
import textstat
import argparse

def calculate_perplexity(text, model, tokenizer):
    # Encode text into input IDs
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # Get maximum length
    max_length = 1024  # 1024
    
    # If text length is less than max length, calculate directly
    if input_ids.size(1) <= max_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.item()
    
    # For long text, calculate in segments
    else:
        # Calculate number of segments needed
        n_segments = (input_ids.size(1) + max_length - 1) // max_length
        perplexities = []
        
        # Calculate perplexity for each segment
        for i in range(n_segments):
            start_idx = i * max_length
            end_idx = min((i + 1) * max_length, input_ids.size(1))
            segment = input_ids[:, start_idx:end_idx]
            
            with torch.no_grad():
                outputs = model(segment, labels=segment)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
        
        # Return average perplexity of all segments
        return sum(perplexities) / len(perplexities)

# Add semantic similarity calculation function
def get_sentence_embedding(texts, model, tokenizer):
    # Encode input
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # Calculate embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Use average of last layer hidden states as sentence representation
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_semantic_similarity(emb1, emb2):
    # Calculate cosine similarity
    return 1 - cosine(emb1.numpy().flatten(), emb2.numpy().flatten())

def calculate_rouge_l(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

# Add Flesch related functions
def calculate_flesch(text):
    return textstat.flesch_reading_ease(text)

def calculate_delta_flesch(before_text, after_text):
    flesch_before = calculate_flesch(before_text)
    flesch_after = calculate_flesch(after_text)
    delta_flesch = abs(flesch_after - flesch_before) / flesch_before * 100
    return delta_flesch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detectLLM', type=str, nargs='+', default=['Moonshot', 'gpt35'], choices=['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini', 'ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'Claude', 'StableLM'])
    parser.add_argument('--dataset', type=str,  nargs='+', default="Physics")
    parser.add_argument('--attack', type=str,  nargs='+', default=['clean' 'raft'])

    args = parser.parse_args()

    models = args.detectLLM
    data_types = args.dataset
    methods = args.attack
    # Load models
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load model for semantic similarity
    semantic_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model = AutoModel.from_pretrained(semantic_model_name)

    # Define file combination parameters
    #models = ['gpt-4omini','gpt35','Llama3','Mixtral','Moonshot']
    #models = ['Llama3']
    #data_types = ['Physics','Math','Computer_science','Biology','Electrical_engineering', 'Statistics','Chemistry','Medicine']
    #data_types = ['Education','Economy','Management']
    #data_types = ['Literature','Law','Art','History','Philosophy']
    #data_types = ['Literature','Law','Art','History','Philosophy']
    #data_types = ['Physics']
    #methods = ['raft']

    # Create results storage list
    results = []

    # Process all combinations
    for model_name in models:
        for data_type in data_types:
            for method in methods:
                if method == 'clean':
                    continue  # Skip clean vs clean comparison
                    
                file_path_1 = f'./pre_dataset/clean_{model_name}_{data_type}.csv'
                #file_path_2 = f'../pre_dataset_0214/{method}_{model_name}_{data_type}.csv'
                file_path_2 = f'./pre_dataset/{method}_{model_name}_{data_type}.csv'
                
                print(f"\nProcessing file pair:")
                print(f"File 1: {file_path_1}")
                print(f"File 2: {file_path_2}")
                
                try:
                    # Add the following function before reading CSV files
                    def print_problematic_line(file_path, line_number):
                        try:
                            with open(file_path, 'r') as file:
                                for i, line in enumerate(file, 1):
                                    if i == line_number:
                                        print(f"\nProblematic line ({line_number}):")
                                        print(line.strip())
                                        break
                        except Exception as e:
                            print(f"Error reading file: {e}")

                    # Modify the CSV file reading part
                    try:
                        # Use stricter CSV parsing parameters
                        df1 = pd.read_csv(file_path_1, 
                                        quoting=csv.QUOTE_ALL,  # Use quotes for all fields
                                        escapechar='\\',        # Use backslash as escape character
                                        engine='python',        # Use Python parsing engine
                                        encoding='utf-8')       # Explicitly specify encoding
                        
                        df2 = pd.read_csv(file_path_2,
                                        quoting=csv.QUOTE_ALL,
                                        escapechar='\\',
                                        engine='python',
                                        encoding='utf-8')
                        
                    except pd.errors.ParserError as e:
                        print(f"\nParsing error: {str(e)}")
                        # Try using more lenient parsing method
                        try:
                            df1 = pd.read_csv(file_path_1, 
                                            quoting=csv.QUOTE_MINIMAL,
                                            engine='python',
                                            on_bad_lines='skip',)
                            
                            df2 = pd.read_csv(file_path_2,
                                            quoting=csv.QUOTE_MINIMAL,
                                            engine='python',
                                            on_bad_lines='skip'
                                            )
                            
                            print("Successfully read files using fault-tolerant mode")
                        except Exception as e2:
                            print(f"Second attempt failed: {str(e2)}")
                            error_line = int(str(e).split("line")[1].split(",")[0])
                            print_problematic_line(file_path_2, error_line)
                            raise

                    # Verify if data was read correctly
                    if 'df1' in locals() and 'df2' in locals():
                        print(f"Original rows in df1: {len(df1)}")
                        print(f"Original rows in df2: {len(df2)}")
                    else:
                        print("File reading failed, program terminated")
                        exit(1)

                    # Reset index and ensure both DataFrames have the same index
                    df1 = df1.reset_index(drop=True)
                    df2 = df2.reset_index(drop=True)

                    # Only keep rows that exist in both DataFrames
                    common_indices = df1.index.intersection(df2.index)
                    df1 = df1.loc[common_indices]
                    df2 = df2.loc[common_indices]

                    print(f"Number of common rows after processing: {len(df1)}")

                    # Modify calculation part
                    original_ppls = []
                    modified_ppls = []
                    semantic_similarities = []
                    rouge_l_scores = []  # New ROUGE-L score list
                    original_flesch = []
                    modified_flesch = []
                    delta_flesch_scores = []

                    for i in range(len(df1)):
                        text1 = df1.iloc[i, 0]
                        text2 = df2.iloc[i, 0]
                        
                        # Check if two texts are exactly the same
                        if text1 == text2:
                            continue
                        
                        # Calculate perplexity
                        ppl1 = calculate_perplexity(text1, model, tokenizer)
                        ppl2 = calculate_perplexity(text2, model, tokenizer)
                        if ppl1>500:
                            print("Strange text:",text1)
                            print("Strange ppl:",ppl1)
                            ppl1=0
                        
                        # Skip nan values
                        if np.isnan(ppl2):
                            print(f"\nFound NaN value at index {i}:")
                            print(f"Original text: {text1[:200]}...")  # Only print first 200 characters
                            print(f"Modified text: {text2[:200]}...")
                            print(f"ppl1: {ppl1}")
                            print(f"ppl2: {ppl2}")
                            continue
                            
                        original_ppls.append(ppl1)
                        modified_ppls.append(ppl2)
                        
                        # Calculate semantic similarity
                        emb1 = get_sentence_embedding(text1, semantic_model, semantic_tokenizer)
                        emb2 = get_sentence_embedding(text2, semantic_model, semantic_tokenizer)
                        similarity = calculate_semantic_similarity(emb1[0], emb2[0])
                        
                        semantic_similarities.append(similarity)
                        
                        # Calculate ROUGE-L score
                        rouge_l = calculate_rouge_l(text1, text2)
                        rouge_l_scores.append(rouge_l)
                        
                        # Add Flesch score calculation
                        flesch1 = calculate_flesch(text1)
                        flesch2 = calculate_flesch(text2)
                        delta_flesch = calculate_delta_flesch(text1, text2)
                        
                        original_flesch.append(flesch1)
                        modified_flesch.append(flesch2)
                        delta_flesch_scores.append(delta_flesch)

                    # Add valid data statistics
                    valid_count = len(original_ppls)
                    total_count = len(df1)
                    filtered_count = total_count - valid_count

                    print(f"\nData Statistics:")
                    print(f"Total data count: {total_count}")
                    print(f"Valid data count: {valid_count}")
                    print(f"Filtered data count: {filtered_count}")

                    # Calculate averages
                    avg_original_ppl = sum(original_ppls) / len(original_ppls)
                    avg_modified_ppl = sum(modified_ppls) / len(modified_ppls)

                    avg_ppl_diff = avg_modified_ppl - avg_original_ppl  # Difference
                    avg_similarity = sum(semantic_similarities) / len(semantic_similarities)
                    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
                    avg_original_flesch = sum(original_flesch) / len(original_flesch)
                    avg_modified_flesch = sum(modified_flesch) / len(modified_flesch)
                    avg_delta_flesch = sum(delta_flesch_scores) / len(delta_flesch_scores)

                    print(f"\nStatistical Information:")
                    print(f"Number of inconsistent texts: {len(original_ppls)}")
                    print(f"Average perplexity of original text: {avg_original_ppl:.4f}")
                    print(f"Average perplexity of modified text: {avg_modified_ppl:.4f}")
                    print(f"Average perplexity difference: {avg_ppl_diff:.4f}")
                    print(f"Average semantic similarity: {avg_similarity:.4f}")
                    print(f"Average ROUGE-L score: {avg_rouge_l:.4f}")
                    print(f"Average Flesch score of original text: {avg_original_flesch:.4f}")
                    print(f"Average Flesch score of modified text: {avg_modified_flesch:.4f}")
                    print(f"Average Flesch score change percentage: {avg_delta_flesch:.4f}%")

                    # Add statistical results to results list
                    result = {
                        'model_name': model_name,
                        'data_type': data_type,
                        'method': method,
                        'total_samples': total_count,
                        'valid_samples': valid_count,
                        'filtered_samples': filtered_count,
                        'original_ppl': avg_original_ppl,
                        'modified_ppl': avg_modified_ppl,
                        'ppl_diff': avg_ppl_diff,
                        'semantic_similarity': avg_similarity,
                        'rouge_l': avg_rouge_l,
                        'original_flesch': avg_original_flesch,
                        'modified_flesch': avg_modified_flesch,
                        'delta_flesch': avg_delta_flesch
                    }
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error processing file pair: {str(e)}")
                    continue

    # Save results to CSV file
    results_df = pd.DataFrame(results)
    # Append to existing CSV file
    #results_df.to_csv('evaluation_results.csv', mode='a', header=False, index=False)
    #print("\nResults have been appended to evaluation_results.csv")
    print(results_df)
