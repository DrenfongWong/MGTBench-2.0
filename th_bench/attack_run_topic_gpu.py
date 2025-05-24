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
import pynvml
import time

categories = ['STEM', 'Humanities', 'Social_sciences']

max_memory_usage = 0
def get_all_gpus_memory_usage_by_pid(pid):
    global max_memory_usage
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    all_gpus_memory_usage = []
    total_memory_usage = 0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        gpu_memory_used = 0
        for proc in processes:
            if proc.pid == pid:
                gpu_memory_used += proc.usedGpuMemory / (1024**2)  # 转换为MiB
        all_gpus_memory_usage.append((i, gpu_memory_used))
        total_memory_usage += gpu_memory_used
    if total_memory_usage > max_memory_usage:
        max_memory_usage = total_memory_usage
    pynvml.nvmlShutdown()
    return all_gpus_memory_usage, total_memory_usage, max_memory_usage

def print_gpu_usage(log_file, category, pid, avg_time_per_input=None):
    gpu_memory_usages, total_memory_usage, max_memory_usage = get_all_gpus_memory_usage_by_pid(pid)
    with open(log_file, 'a') as f:
        f.write(f"Category: {category}\n")
        f.write(f"PID: {pid}\n")
        for gpu_id, memory_usage in gpu_memory_usages:
            f.write(f"GPU {gpu_id} Memory Usage: {memory_usage:.2f} MiB\n")
        f.write(f"Total GPU Memory Usage: {total_memory_usage:.2f} MiB\n")
        f.write(f"Max GPU Memory Usage: {max_memory_usage:.2f} MiB\n")
        if avg_time_per_input is not None:
            f.write(f"Average Time per Input: {avg_time_per_input:.4f} seconds\n")
        f.write("\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--detectLLM', type=str, default='Moonshot', choices=['Moonshot', 'gpt35', 'Mixtral', 'Llama3', 'gpt-4omini', 'ChatGPT-turbo', 'ChatGLM', 'Dolly', 'ChatGPT', 'GPT4All', 'Claude', 'StableLM'])
    parser.add_argument('--dataset', type=str, default="AITextDetect")
    parser.add_argument('--attack', type=str, default='clean')
    parser.add_argument('--datatype', type=str, default='mgt2')


    args = parser.parse_args()

    datasource = args.dataset

    datatype = args.datatype
    if datatype == "mgt1":
        categories = ['Essay', 'Reuters', 'WP']
    elif datatype == "mgt2":
        categories = ['STEM', 'Humanities', 'Social_sciences']        

    save_dir =  "pre_dataset"
    os.makedirs(save_dir, exist_ok=True)  # 动态创建目录（如果不存在）
    gpu_log_file = os.path.join(save_dir, f"gpu_usage_{args.detectLLM}_{args.attack}.log")
    current_pid = os.getpid()
    # 每1秒记录一次GPU使用情况
    def monitor_gpu():
        while True:
            print_gpu_usage(gpu_log_file, "Global" , current_pid)
            time.sleep(1)
            
    import threading
    gpu_monitor_thread = threading.Thread(target=monitor_gpu)
    gpu_monitor_thread.daemon = True  # 设置为守护线程，主程序结束时自动退出
    gpu_monitor_thread.start()
    # run all the experiments
    cnt = 0
    for cat in categories:
        for length in range(100, 1100, 100):
            max_memory_usage = 0
            start_time = time.time()
            with open(gpu_log_file, 'a') as f:
                f.write(f"Category: {cat}_{length}\n")
                f.write("\n")  # 添加一个空行以便区分不同的记录    
            file_name = f"clean_{length}_{args.detectLLM}_{cat}.csv"
            file_path = os.path.join(save_dir, file_name)
            df = pd.read_csv(file_path)
            data = {'train': {
                        'text': [], 
                        'label': [], }, 
                    'test': { 
                        'text': [], 
                        'label': [], }
                    }
            data['test']['text'] = df['text'].tolist()
            data['test']['label'] = df['label'].tolist()
            if args.attack == "dipper":
                data['test'] = run_attack_dipper(data=data['test'])
            elif args.attack == "recursive_dipper":
                data['test'] = run_attack_dipper(data=data['test'], P=5)
            elif args.attack == "token_ensemble":
                data['test'] = run_token_ensemble(data=data['test'])
            elif args.attack == "token_ensemble_with_ppl":
                data['test'] = run_token_ensemble(data=data['test'])
            elif args.attack == "raft":
                experiment = Experiment(
                    data['test'],
                    f"{length}_{args.attack}_{args.detectLLM}_{cat}",
                    "llama-2-11b-chat",
                    "gpt2",
                    "roberta-base",
                    "./experiments/",
                    "cuda",
                    "cuda",
                    0.1,
                    15,
                    "gpt-3.5-turbo",
                    "./exp_gpt3to4/data/",
                )
                data['test'] = experiment.run()
            end_time = time.time()  # 记录处理结束时间
            total_time = end_time - start_time # 计算总运行时间
            avg_time_per_input = total_time / len(df) if len(df) > 0 else 0  # 计算每条输入的平均运行时间
            print(f"Total Time: {total_time:.4f} seconds")
            print(f"Average Time per Input: {avg_time_per_input:.4f} seconds")
            # 调用 print_gpu_usage 记录 GPU 使用情况和平均运行时间
            print_gpu_usage(gpu_log_file, "Final", current_pid, avg_time_per_input)
            print(f'===== {cat} - {args.detectLLM} - {args.attack} - {length}=====')
            combined_data = data['test']  # 假设数据是列表形式
            with open(f'pre_dataset/{args.attack}_{length}_{args.detectLLM}_{cat}.json', "w") as f:
                json.dump(data, f, indent=4)
            file_name = f"{args.attack}_{length}_{args.detectLLM}_{cat}.csv"
            file_path = os.path.join(save_dir, file_name)
            df = pd.DataFrame(combined_data)
            df.to_csv(file_path, index=False)
            print(f"数据已成功保存！文件路径：{file_path}")
