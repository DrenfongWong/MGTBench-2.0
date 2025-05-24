import os
import json
import logging
import argparse
import torch
import torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
import pynvml
import time
import tqdm
from datetime import datetime
from textflint.adapter import auto_dataset
from textattack.shared import AttackedText
from textattack.goal_function_results import GoalFunctionResultStatus

from attack.methods.models import SurrogateDetectionModel
from utils.conf_util import setup_logger


attacking = None

logger = logging.getLogger()
setup_logger(logger)

max_memory_usage = 0
current_pid = 0
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

def init(model_path, attacking_method="dualir", gpu_i=0, gpu_log_file=None):
    global attacking

    if model_path is None:
       raise ValueError("model_path cannot be None")
    os.environ["VICTIM_DEVICE"] = f"cuda:{gpu_i}"
    os.environ["PPL_DEVICE"] = f"cuda:{gpu_i}"
    os.environ["TA_DEVICE"] = f"cuda:{gpu_i}"
    from textattack.shared import utils
    utils.device = f"cuda:{gpu_i}"
    
    if attacking_method == "dualir":
        from attack.recipes.rspmu_mlm_dualir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "wir":
        from attack.recipes.rspmu_mlm_wir import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "greedy":
        from attack.recipes.rspmu_mlm_greedy import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_pos":
        from attack.recipes.ablation_rsmu_mlm_dualir_no_pos import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_use":
        from attack.recipes.ablation_rspm_mlm_dualir_no_use import get_recipe
        recipe_func = get_recipe
    elif attacking_method == "no_max_perturbed":
        from attack.recipes.ablation_rspu_mlm_dualir_no_max_perturbed import get_recipe
        recipe_func = get_recipe
    else:
        raise NotImplementedError(f"not supported attacking recipe -> {attacking_method}")

    # init model specific arguments
    # TODO, we set human class index to 1 in early surrogate model training on CheckGPT
    label2id = {"human": 1, "gpt": 0, "tied": 0}
    target_cls = 1

    victim_model = SurrogateDetectionModel(model_path, batch_size=128, label2id=label2id)
    attacking = recipe_func(target_cls)
    attacking.init_goal_function(victim_model)

    logger.info(f"run attacking with {recipe_func}")
    logger.info(
        "*******************************\n"
        f"initializing process...]\n"
        f"\t gpus: {os.environ.get('CUDA_CUDA_VISIBLE_DEVICES', None)}\n"
        f"\t victim on {os.environ.get('VICTIM_DEVICE', None)}\n"
        f"\t ppl on {os.environ.get('PPL_DEVICE', None)}\n"
        f"\t textattack on {os.environ.get('TA_DEVICE', None)}\n"
        f"\t attacking recipe: \n"
        f"{attacking.print()}\n"
        "*******************************\n",
        # flush=True,
    )
    global current_pid
    current_pid = os.getpid()
    # 每10秒记录一次GPU使用情况
    def monitor_gpu():
        while True:
            print_gpu_usage(gpu_log_file, "Global" , current_pid)
            time.sleep(1)
    import threading
    gpu_monitor_thread = threading.Thread(target=monitor_gpu)
    gpu_monitor_thread.daemon = True  # 设置为守护线程，主程序结束时自动退出
    gpu_monitor_thread.start()


class MultiProcessingHelper:
    def __init__(self):
        self.total = None
    def __call__(self, data_samples, trans_save_path, func, workers=None, init_fn=None, init_args=None):
        self.total = len(data_samples)
        results = []
        with mp.Pool(workers, initializer=init_fn, initargs=init_args) as pool, \
             tqdm.tqdm(pool.imap(func, data_samples), total=self.total, dynamic_ncols=True) as pbar, \
             open(trans_save_path, "wt") as w_trans:
                for trans_res in pbar:
                    if trans_res is None: continue
                    w_trans.write(json.dumps(trans_res.dump(), ensure_ascii=False) + "\n")
                    results.append(trans_res.dump())
        return results

def init_sample_from_textattack(ori):
    text_input, label_str = ori.to_tuple()
    label_output = attacking.goal_function.model.label2id[label_str]
    attacked_text = AttackedText(text_input)
    print ("attacking: label, text:", label_output, label_str, attacked_text,"\n\n\n")
    if attacked_text.num_words <= 2:
        logger.debug(f"\n\n\n\n\n\n\n\n\n\nThe initial text -> [{attacked_text.text}] is less than 2 words, will skip this sample now!!!")
        goal_function_result = None
    else:
        goal_function_result, _ = attacking.goal_function.init_attack_example(
            attacked_text, label_output
        )
    print
    return goal_function_result


def do_attack_one(ori_one):
    goal_function_result = init_sample_from_textattack(ori_one)

    if goal_function_result is None or \
        goal_function_result.goal_status == GoalFunctionResultStatus.SKIPPED:
        print ("skip skip skip \n\n\n\n\n\n\n\n\n")
        return None
    else:
        result = attacking.attack_one(goal_function_result)
        train_data_dict = result.perturbed_result.attacked_text._text_input
        trans_sample = ori_one.replace_fields(
            list(train_data_dict.keys()),
            list(train_data_dict.values()),
        )
        return trans_sample


def run_flint_attack(args):
    start_time = time.time()
    # prepare test data
    import pandas as pd
    save_dir =  "pre_dataset"
    file_name = f"clean_{args.model_dataset}.csv"
    file_path = os.path.join(save_dir, file_name)
    df = pd.read_csv(file_path)
    data = {} 
    data['text'] = df['text'].tolist()
    data['label'] = df['label'].tolist()
    sample_list = list()
    for idx, dd in tqdm.tqdm(enumerate(data['text']), total=len(data['text'])):
        if data['label'][idx] == 1:
            sample_list.append({
                "x": data["text"][idx],
                "y": 'gpt',
            })
        elif data['label'][idx] == 0:
            sample_list.append({
                "x": data["text"][idx],
                "y": 'human',
            })
    print (sample_list)
    dataset = auto_dataset(sample_list, task="SA")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    output_file = os.path.join(
        args.output_dir,
        "attacked-{}-{}.jsonl".format(args.attacking_method, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    )
    gpu_log_file = os.path.join(args.output_dir, f"gpu_usage_{args.model_dataset}.log")
    worker = MultiProcessingHelper()
    results = worker(
        dataset,
        output_file, 
        func=do_attack_one,
        workers=1,
        init_fn=init,
        init_args=(args.model_name_or_path, args.attacking_method, 1, gpu_log_file,),
    )
    end_time = time.time()  # 记录处理结束时间
    total_time = end_time - start_time  # 计算总运行时间
    avg_time_per_input = total_time / len(df) if len(df) > 0 else 0  # 计算每条输入的平均运行时间
    print(f"Total Time: {total_time:.4f} seconds")
    print(f"Average Time per Input: {avg_time_per_input:.4f} seconds")
            # 调用 print_gpu_usage 记录 GPU 使用情况和平均运行时间
    global current_pid
    print ("current_pid", current_pid)
    print_gpu_usage(gpu_log_file, "Final", current_pid, avg_time_per_input)
    for idx, dd in tqdm.tqdm(enumerate(results), total=len(results)):
        data['text'][results[idx]['sample_id']] = results[idx]['x']
    
    with open(f'{args.output_dir}/flint_attack_{args.model_dataset}.json', "w") as f:
        json.dump(data, f, indent=4)
    df = pd.DataFrame(data)
    df.to_csv(f'{args.output_dir}/flint_attack_{args.model_dataset}.csv', index=False)
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A simple argument parser example")
    parser.add_argument("--model_name_or_path", required=True, help="TextFlint Model path")
    parser.add_argument("--data_file", required=True, help="TextFlint test file path")
    parser.add_argument("--output_dir", required=True, help="Directory to save the attacked samples")

    parser.add_argument("--attacking_method", type=str, default="dualir", help="Attacking method")
    parser.add_argument("--num_gpu_per_process", type=int, default=1, help="Number of gpus of one process")
    parser.add_argument("--num_workers", type=int, default=1, help="Total gpu usage is num_workers * num_gpu_per_process")
    parser.add_argument("--text_key", type=str, default="text", help="Text key for json object")
    parser.add_argument("--label_key", type=str, default="label", help="Label key for json object")
    parser.add_argument("--model_dataset", type=str, default="model_dataset", help="attack model and dataset name")
    args = parser.parse_args()
    run_flint_attack(args)
