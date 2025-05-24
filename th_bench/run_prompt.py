import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import pynvml
import time
import os
import torch

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

def generate_variation(prompt, input_text, model, tokenizer, device):
    full_input = f"{prompt} {input_text}"
    inputs = tokenizer(full_input, return_tensors="pt").to(device)
    with torch.no_grad():  # 禁用梯度计算
        output = model.generate(**inputs, max_length=2048)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

def process_csv(file_path, model_path, prompt):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    df = pd.read_csv(file_path)
    results = []
    start_time = time.time()
    for index, row in df.iterrows():
        input_text = row['text']
        generated_text = generate_variation(prompt, input_text, model, tokenizer, device)
        results.append(generated_text)
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_per_input = total_time / len(df)
    output_df = pd.DataFrame(results, columns=['generated_variation'])
    output_df.to_csv('output.csv', index=False)
    print("Generated variations saved to output.csv")
    return avg_time_per_input

def main():
    parser = argparse.ArgumentParser(description="Generate variations of instructions using a local Llama model.")
    parser.add_argument("--file", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--model", type=str, required=True, help="Path to the local Llama model directory.")
    parser.add_argument("--prompt", type=str, default="Generate a variation of the following instruction while keeping the semantic meaning:", help="Prompt to use for generating variations.")
    args = parser.parse_args()
    current_pid = os.getpid()

    # 根据输入的 CSV 文件路径生成日志文件路径
    log_file = args.file.replace('clean', 'gpu_usage_prompt').replace('.csv', '.log')

    def monitor_gpu():
        while True:
            print_gpu_usage(log_file, "Global", current_pid)
            time.sleep(1)

    import threading
    gpu_monitor_thread = threading.Thread(target=monitor_gpu)
    gpu_monitor_thread.daemon = True
    gpu_monitor_thread.start()

    avg_time_per_input = process_csv(args.file, args.model, args.prompt)
    print_gpu_usage(log_file, "Final", current_pid, avg_time_per_input)

if __name__ == "__main__":
    main()
