import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BertTokenizerFast,get_linear_schedule_with_warmup
import swanlab

from dataset import NERDataset, DataCollatorForNER
from model import BERTNER
from utils import load_config,get_label_map,build_label_file
from trainer import Trainer
from utils import set_seed

def parse_args():
    parser = argparse.ArgumentParser(description="BERT‑NER训练脚本")
    parser.add_argument("--config", type=str, default="configs/msra_bert.json",
                        help="json配置文件路径")

    args = parser.parse_args()
    return args
  
def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    
    swanlab.init(
        project="demo2",
        experiment_name=cfg["experiment_name"],
        config=cfg,   
        logdir=cfg["log_path"]
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    label2id, id2label = get_label_map(cfg["dataset_name"])
    num_labels = len(label2id)
    cfg["num_labels"] = num_labels
    
    tokenizer = BertTokenizerFast.from_pretrained(cfg["model_name"])
    
    train_dataset = NERDataset(cfg, mode="train")
    dev_dataset = NERDataset(cfg, mode="dev")
    test_dataset = NERDataset(cfg, mode="test")
    
    print(f"训练集样本数：{len(train_dataset)}")
    print(f"验证集样本数：{len(dev_dataset)}")
    print(f"测试集样本数：{len(test_dataset)}")
    
    collate_fn = DataCollatorForNER(
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=cfg["max_length"]
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BERTNER(cfg).to(device)
    
    optimizer = AdamW(
        model.parameters(), 
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"]
    )
    
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(cfg["warmup_ratio"] * total_steps)
        
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    
    trainer = Trainer(
        model,
        optimizer,
        scheduler,
        criterion,
        device,
        save_path=cfg["save_path"],
        id2label = id2label,
        patience = 3
    )
    
    for epoch in range(cfg["epochs"]):
        print(f"Epoch {epoch+1}/{cfg['epochs']}")
        train_loss, train_p, train_r,train_f1 = trainer.train_epoch(train_loader)
        print(f"Train Loss: {train_loss:.4f}, P: {train_p:.4f}, R: {train_r:.4f}, F1: {train_f1:.4f}")
        dev_loss, dev_p, dev_r, dev_f1, _, _, early_stop = trainer.eval_epoch(dev_loader,is_dev=True)
        print(f"Dev Loss: {dev_loss:.4f}, P: {dev_p:.4f}, R: {dev_r:.4f}, F1: {dev_f1:.4f}")
    
        swanlab.log({
            "train/loss": train_loss,
            "train/precision": train_p,
            "train/recall": train_r,
            "train/f1": train_f1,
            "dev/loss": dev_loss,
            "dev/precision": dev_p,
            "dev/recall": dev_r,
            "dev/f1": dev_f1
        }, step=epoch + 1)
        
        if early_stop:
            print("早停触发，停止训练。")
            break
    
        
    print("\n==== 训练完成，加载最优模型评估测试集 ====")
    checkpoint = torch.load(cfg["save_path"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_p, test_r, test_f1, _, _ = trainer.test(test_loader)
    print(f"Test -> P:{test_p:.4f}, R:{test_r:.4f}, F1:{test_f1:.4f}")

    swanlab.log({
        "test/precision": test_p,
        "test/recall": test_r,
        "test/f1": test_f1
    })
    
    swanlab.finish()
    
    
if __name__ == "__main__":
    # 预处理标签 运行一次
    # build_label_file("weibo", "data/weibo/train.txt", dev_path="data/weibo/dev.txt", test_path="data/weibo/test.txt")
    # build_label_file("msra", "data/msra/train.txt", dev_path="data/msra/dev.txt", test_path="data/msra/test.txt")

    # 读取验证一下
    # label2id, id2label = get_label_map("weibo")
    # print(label2id)
    main()
    