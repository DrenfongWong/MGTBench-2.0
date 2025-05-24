import torch
import json
import argparse
import os
import pandas as pd
from paraphrase import run_attack_dipper
from dipper_and_raft import run_attack_dipper_and_raft
from token_ensemble import run_token_ensemble
from experiment import Experiment
from dataloader import load

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detectLLM', type=str, default='Moonshot', choices=['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini', 'ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'Claude', 'StableLM'])
    parser.add_argument('--dataset', type=str, default="AITextDetect")
    parser.add_argument('--attack', type=str, default='clean')
    parser.add_argument('--datatype', type=str, default='mgt2')

    args = parser.parse_args()

    datasource = args.dataset

    datatype = args.datatype

    save_dir =  "pre_dataset"
    os.makedirs(save_dir, exist_ok=True)  # 动态创建目录（如果不存在）
    
    if datatype == "mgt1":
        categories = ['Essay', 'Reuters', 'WP']
    elif datatype == "mgt2":
        categories = ['Physics', 'Medicine', 'Biology', 'Electrical_engineering', 'Computer_science',
                    'Literature', 'History', 'Education', 'Art', 'Law', 'Management', 'Philosophy',
                                'Economy', 'Math', 'Statistics', 'Chemistry']
    elif datatype == "mgt2_topic":
        categories = categories = ['STEM', 'Humanities', 'Social_sciences']
    # run all the experiments
    cnt = 0
    for cat in categories:
        file_name = f"clean_{args.detectLLM}_{cat}.csv"
        file_path = os.path.join(save_dir, file_name)
        data = {
            'train': {
                'text': [],
                'label': [],
                    },
            'test': {
                'text': [],
                'label': [],
                    }
                }
        data_new = {
            'train': {
                'text': [],
                'label': [],
                    },
            'test': {
                'text': [],
                'label': [],
                    }
                }
        if args.attack == "clean":
            if datatype == "mgt1":
                data = load(name=cat, detectLLM=args.detectLLM, category=cat)
            elif datatype == "mgt2":
                data = load(name=datasource, detectLLM=args.detectLLM, category=cat)
            for length in range(100, 1100, 100):  # 生成100到1000字长的文本
                clean_texts = []
                clean_labels = []
                count = 0
                for text, label in zip(data['test']['text'], data['test']['label']):
                    words = text.split()
                    if label == 1 and len(words) >= length and len(words) < (length + 100):
                        print (len(words))
                        clean_texts.append(' '.join(words[:length]))
                        clean_labels.append(label)
                        count += 1
                    if count == 10:  # 限制为10条
                        break
                data_new['test'][f'text_{length}'] = clean_texts
                data_new['test'][f'label_{length}'] = clean_labels

            # 打印或保存处理后的数据
            for length in range(100, 1100, 100):
                print(f"Category: {cat}, Length: {length}, Number of texts: {len(data_new['test'][f'text_{length}'])}")

            # 保存处理后的数据到文件
            for length in range(100, 1100, 100):
                output_file_name = f"clean_{length}_{args.detectLLM}_{cat}.csv"
                output_file_path = os.path.join(save_dir, output_file_name)
                clean_df = pd.DataFrame({
                    'text': data_new['test'][f'text_{length}'],
                    'label': data_new['test'][f'label_{length}']
                })
                clean_df.to_csv(output_file_path, index=False)
                print(f"Saved clean text of length {length} to {output_file_path}")
        else:
            df = pd.read_csv(file_path)
            data['test']['text'] = df['text'].tolist()
            data['test']['label'] = df['label'].tolist()
        if args.attack == "dipper":
            data['test'] = run_attack_dipper(data=data['test'])
        elif args.attack == "dipper_and_raft":
            data['test'] = run_attack_dipper_and_raft(data=data['test'])
        elif args.attack == "recursive_dipper":
            data['test'] = run_attack_dipper(data=data['test'], P=5)
        elif args.attack == "token_ensemble":
            data['test'] = run_token_ensemble(data=data['test'])
        elif args.attack == "raft":
            experiment = Experiment(
                data['test'],
                f"{args.attack}_{args.detectLLM}_{cat}",
                "llama-2-11b-chat",
                "gpt2",
                "roberta-base",
                "./experiments/",
                "cuda:1",
                "cuda:2",
                0.1,
                15,
                "gpt-3.5-turbo",
                "./exp_gpt3to4/data/",
            )
            data['test'] = experiment.run()
        print(f'===== {cat} - {args.detectLLM} - {args.attack}=====')
        combined_data = data['test']  # 假设数据是列表形式
        with open(f'pre_dataset/{args.attack}_{args.detectLLM}_{cat}.json', "w") as f:
            json.dump(data, f, indent=4)
        file_name = f"{args.attack}_{args.detectLLM}_{cat}.csv"
        file_path = os.path.join(save_dir, file_name)
        df = pd.DataFrame(combined_data)
        df.to_csv(file_path, index=False)
        print(f"数据已成功保存！文件路径：{file_path}")
