import json
import os
import argparse

def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str,
        default="configs/msra_bert.json")
    args = parser.parse_args()
    config_path = args.config
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
        
    return config

