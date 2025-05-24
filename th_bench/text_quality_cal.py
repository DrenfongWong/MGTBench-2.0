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
    # 将文本编码为输入ID
    input_ids = tokenizer.encode(text, return_tensors='pt')
    
    # 获取最大长度
    max_length = 1024  # 1024
    
    # 如果文本长度小于最大长度，直接计算
    if input_ids.size(1) <= max_length:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            perplexity = torch.exp(loss)
            return perplexity.item()
    
    # 对于长文本，分段计算
    else:
        # 计算需要多少段
        n_segments = (input_ids.size(1) + max_length - 1) // max_length
        perplexities = []
        
        # 对每一段计算困惑度
        for i in range(n_segments):
            start_idx = i * max_length
            end_idx = min((i + 1) * max_length, input_ids.size(1))
            segment = input_ids[:, start_idx:end_idx]
            
            with torch.no_grad():
                outputs = model(segment, labels=segment)
                loss = outputs.loss
                perplexity = torch.exp(loss)
                perplexities.append(perplexity.item())
        
        # 返回所有段的平均困惑度
        return sum(perplexities) / len(perplexities)

# 添加语义相似度计算函数
def get_sentence_embedding(texts, model, tokenizer):
    # 对输入进行编码
    encoded_input = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
    
    # 计算embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        # 使用最后一层隐藏状态的平均值作为句子表示
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    
    return sentence_embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def calculate_semantic_similarity(emb1, emb2):
    # 计算余弦相似度
    return 1 - cosine(emb1.numpy().flatten(), emb2.numpy().flatten())

def calculate_rouge_l(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(text1, text2)
    return scores['rougeL'].fmeasure

# 添加 Flesch 相关函数
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
    # 加载模型
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # 加载用于语义相似度的模型
    semantic_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    semantic_tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)
    semantic_model = AutoModel.from_pretrained(semantic_model_name)

    # 定义文件组合参数
    #models = ['gpt-4omini','gpt35','Llama3','Mixtral','Moonshot']
    #models = ['Llama3']
    #data_types = ['Physics','Math','Computer_science','Biology','Electrical_engineering', 'Statistics','Chemistry','Medicine']
    #data_types = ['Education','Economy','Management']
    #data_types = ['Literature','Law','Art','History','Philosophy']
    #data_types = ['Literature','Law','Art','History','Philosophy']
    #data_types = ['Physics']
    #methods = ['raft']

    # 创建结果存储列表
    results = []

    # 循环处理所有组合
    for model_name in models:
        for data_type in data_types:
            for method in methods:
                if method == 'clean':
                    continue  # 跳过clean vs clean的对比
                    
                file_path_1 = f'./pre_dataset/clean_{model_name}_{data_type}.csv'
                #file_path_2 = f'../pre_dataset_0214/{method}_{model_name}_{data_type}.csv'
                file_path_2 = f'./pre_dataset/{method}_{model_name}_{data_type}.csv'
                
                print(f"\n处理文件对：")
                print(f"File 1: {file_path_1}")
                print(f"File 2: {file_path_2}")
                
                try:
                    # 在读取CSV文件之前添加以下函数
                    def print_problematic_line(file_path, line_number):
                        try:
                            with open(file_path, 'r') as file:
                                for i, line in enumerate(file, 1):
                                    if i == line_number:
                                        print(f"\n问题行 ({line_number}):")
                                        print(line.strip())
                                        break
                        except Exception as e:
                            print(f"读取文件时出错: {e}")

                    # 修改读取CSV文件的部分
                    try:
                        # 使用更严格的CSV解析参数
                        df1 = pd.read_csv(file_path_1, 
                                        quoting=csv.QUOTE_ALL,  # 使用引号包围所有字段
                                        escapechar='\\',        # 使用反斜杠作为转义字符
                                        engine='python',        # 使用Python解析引擎
                                        encoding='utf-8')       # 明确指定编码
                        
                        df2 = pd.read_csv(file_path_2,
                                        quoting=csv.QUOTE_ALL,
                                        escapechar='\\',
                                        engine='python',
                                        encoding='utf-8')
                        
                    except pd.errors.ParserError as e:
                        print(f"\n解析错误: {str(e)}")
                        # 尝试使用更宽松的解析方式
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
                            
                            print("使用容错模式成功读取文件")
                        except Exception as e2:
                            print(f"第二次尝试失败: {str(e2)}")
                            error_line = int(str(e).split("line")[1].split(",")[0])
                            print_problematic_line(file_path_2, error_line)
                            raise

                    # 验证数据是否正确读取
                    if 'df1' in locals() and 'df2' in locals():
                        print(f"df1原始行数: {len(df1)}")
                        print(f"df2原始行数: {len(df2)}")
                    else:
                        print("文件读取失败，程序终止")
                        exit(1)

                    # 重置索引并确保两个DataFrame有相同的索引
                    df1 = df1.reset_index(drop=True)
                    df2 = df2.reset_index(drop=True)

                    # 只保留两个DataFrame都存在的行
                    common_indices = df1.index.intersection(df2.index)
                    df1 = df1.loc[common_indices]
                    df2 = df2.loc[common_indices]

                    print(f"处理后共同行数: {len(df1)}")

                    # 修改计算部分
                    original_ppls = []
                    modified_ppls = []
                    semantic_similarities = []
                    rouge_l_scores = []  # 新增ROUGE-L分数列表
                    original_flesch = []
                    modified_flesch = []
                    delta_flesch_scores = []

                    for i in range(len(df1)):
                        text1 = df1.iloc[i, 0]
                        text2 = df2.iloc[i, 0]
                        
                        # 检查两个文本是否完全一致
                        if text1 == text2:
                            continue
                        
                        # 计算困惑度
                        ppl1 = calculate_perplexity(text1, model, tokenizer)
                        ppl2 = calculate_perplexity(text2, model, tokenizer)
                        if ppl1>500:
                            print("strange text:",text1)
                            print("strange ppl:",ppl1)
                            ppl1=0
                        
                        # 跳过 nan 值
                        if np.isnan(ppl2):
                            print(f"\n发现 NaN 值在索引 {i}:")
                            print(f"原始文本: {text1[:200]}...")  # 只打印前200个字符
                            print(f"修改后文本: {text2[:200]}...")
                            print(f"ppl1: {ppl1}")
                            print(f"ppl2: {ppl2}")
                            continue
                            
                        original_ppls.append(ppl1)
                        modified_ppls.append(ppl2)
                        
                        # 计算语义相似度
                        emb1 = get_sentence_embedding(text1, semantic_model, semantic_tokenizer)
                        emb2 = get_sentence_embedding(text2, semantic_model, semantic_tokenizer)
                        similarity = calculate_semantic_similarity(emb1[0], emb2[0])
                        
                        semantic_similarities.append(similarity)
                        
                        # 计算ROUGE-L分数
                        rouge_l = calculate_rouge_l(text1, text2)
                        rouge_l_scores.append(rouge_l)
                        
                        # 添加 Flesch 分数计算
                        flesch1 = calculate_flesch(text1)
                        flesch2 = calculate_flesch(text2)
                        delta_flesch = calculate_delta_flesch(text1, text2)
                        
                        original_flesch.append(flesch1)
                        modified_flesch.append(flesch2)
                        delta_flesch_scores.append(delta_flesch)

                    # 添加有效数据统计
                    valid_count = len(original_ppls)
                    total_count = len(df1)
                    filtered_count = total_count - valid_count

                    print(f"\n数据统计：")
                    print(f"总数据条数：{total_count}")
                    print(f"有效数据条数：{valid_count}")
                    print(f"被过滤数据条数：{filtered_count}")

                    # 计算平均值
                    avg_original_ppl = sum(original_ppls) / len(original_ppls)
                    avg_modified_ppl = sum(modified_ppls) / len(modified_ppls)

                    avg_ppl_diff = avg_modified_ppl - avg_original_ppl  # 差值
                    avg_similarity = sum(semantic_similarities) / len(semantic_similarities)
                    avg_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)
                    avg_original_flesch = sum(original_flesch) / len(original_flesch)
                    avg_modified_flesch = sum(modified_flesch) / len(modified_flesch)
                    avg_delta_flesch = sum(delta_flesch_scores) / len(delta_flesch_scores)

                    print(f"\n统计信息：")
                    print(f"不一致文本数量：{len(original_ppls)}")
                    print(f"原始文本平均困惑度：{avg_original_ppl:.4f}")
                    print(f"修改后文本平均困惑度：{avg_modified_ppl:.4f}")
                    print(f"困惑度平均差值：{avg_ppl_diff:.4f}")
                    print(f"平均语义相似度：{avg_similarity:.4f}")
                    print(f"平均ROUGE-L分数：{avg_rouge_l:.4f}")
                    print(f"原始文本平均 Flesch 分数：{avg_original_flesch:.4f}")
                    print(f"修改后文本平均 Flesch 分数：{avg_modified_flesch:.4f}")
                    print(f"Flesch 分数平均变化比例：{avg_delta_flesch:.4f}%")

                    # 将统计结果添加到结果列表
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
                    print(f"处理文件对时出错: {str(e)}")
                    continue

    # 将结果保存为CSV文件
    results_df = pd.DataFrame(results)
    # 追加到已有的CSV文件
    #results_df.to_csv('evaluation_results.csv', mode='a', header=False, index=False)
    #print("\n结果已追加到 evaluation_results.csv")
    print(results_df)
