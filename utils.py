import os
import json
import random
import numpy as np
import torch

path = "configs/labels.json"

def load_config(config_path):
   
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        
    return cfg

def extract_labels(data_path):
    labels = set()

    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, label = line.split()
            labels.add(label)
    return labels

def save_labels(dataset_name, labels):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_labels = json.load(f)
    else:
        all_labels = {}            
    all_labels[dataset_name] = labels
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(all_labels, f, ensure_ascii=False, indent=4)


def build_label_file(dataset_name,train_path, dev_path=None, test_path=None):
    labels = extract_labels(train_path)
    for file_path in [dev_path, test_path]:
        if file_path is not None:
            labels.update(extract_labels(file_path))
    labels = sorted(labels)
    if "O" in labels:
        labels.remove("O")
        labels = ["O"] + labels
    save_labels(dataset_name, labels)


def load_labels(dataset_name):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            all_labels = json.load(f)
        if dataset_name in all_labels:
            return all_labels[dataset_name]
    return None

def get_label_map(dataset_name):
    
    labels = load_labels(dataset_name)
    if labels is None:
        raise ValueError(f"Labels for dataset '{dataset_name}' not found.")
    
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    return label2id, id2label

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
    
    
if __name__ == "__main__":
    build_label_file("weibo", "data/weibo/train.txt")
    build_label_file("msra", "data/MSRA/train.txt")
    weibo_label2id, weibo_id2label = get_label_map("weibo")
    print("weibo label2id：", weibo_label2id)
    print("weibo id2label：", weibo_id2label)