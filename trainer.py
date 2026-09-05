import os
import torch
from tqdm import tqdm
from metric import Metric

class Trainer:
    def __init__(self, model,opt,scheduler,device,save_path,id2label,cfg,patience=3):
        self.model = model
        self.opt = opt
        self.scheduler = scheduler
        self.device = device
        self.save_path = save_path
        self.id2label = id2label
        self.cfg = cfg
        self.best_dev_f1 = float("-inf")
        
        self.patience = patience      
        self.early_stop_counter = 0   
        self.early_stop = False
        self.current_epoch = 0
        
    def early_stop_check(self, current_f1):
        if current_f1 >= self.best_dev_f1:
            self.best_dev_f1 = current_f1
            self.early_stop_counter = 0
            checkpoint = {
                "model_state_dict": self.model.state_dict(),       
                "optimizer_state_dict": self.opt.state_dict(),     
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_f1": float(self.best_dev_f1),
                "id2label": self.id2label,
                "config": self.cfg,
                "epoch": self.current_epoch,
                "dataset_name": self.cfg.get("dataset_name", "unknown"),
                "lr": self.cfg.get("lr", -1),
                "batch_size": self.cfg.get("batch_size", -1)
            }
            save_dir = os.path.dirname(self.save_path)
            if save_dir:  
                os.makedirs(save_dir, exist_ok=True)
            torch.save(checkpoint, self.save_path)
            print(f"更新最优模型, best_f1:{self.best_dev_f1:.4f}, path:{self.save_path}")
        else:
            self.early_stop_counter += 1
            print(f"dev集F1无提升，早停计数器: {self.early_stop_counter}/{self.patience}")
            if self.early_stop_counter >= self.patience:
                self.early_stop = True
                print(f"连续{self.patience}轮验证集F1未提升，停止训练。最优F1={self.best_dev_f1:.4f}")
        return self.early_stop
    
    def _decode_batch(self, logits, labels):
        pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
        true_ids = labels.detach().cpu().numpy()
        
        pred_tags = []
        true_tags = []
        
        for p_row, t_row in zip(pred_ids, true_ids):
            p_tags = []
            t_tags = []
            for p, t in zip(p_row, t_row):
                if t != -100:  
                    p_tags.append(self.id2label[p])
                    t_tags.append(self.id2label[t])
            pred_tags.append(p_tags)
            true_tags.append(t_tags)
        
        return pred_tags, true_tags
    
    def train_epoch(self,loader):
        self.model.train()
        total_loss = 0.0
        
        for batch in tqdm(loader, desc="Train"):
            input_ids = batch["input_ids"].to(self.device)
            mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            self.opt.zero_grad()

            out = self.model(input_ids, mask, labels)
            loss = out["loss"]
            
            loss.backward()
            self.opt.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(loader)    

        return avg_loss
        
    def eval_epoch(self,loader,is_dev):
        self.model.eval() 
        total_loss = 0.0
        pred_tags = []
        true_tags = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Eval"):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                out = self.model(input_ids, mask, labels)
                loss = out["loss"]
                logits = out["logits"]
                total_loss += loss.item()

                batch_pred, batch_true = self._decode_batch(logits, labels)
                pred_tags.extend(batch_pred)
                true_tags.extend(batch_true)
                     
        avg_loss = total_loss / len(loader)
        metric = Metric(true_tags, pred_tags)
        p, r, f1 = metric.precision, metric.recall, metric.f1
                
        
        if is_dev :
            self.early_stop_check(f1)
            
        return avg_loss, p, r, f1, pred_tags, true_tags,self.early_stop
    
    def test(self,loader):
        self.model.eval() 
        pred_tags = []
        true_tags = []
        
        with torch.no_grad():
            for batch in tqdm(loader, desc="Test"):
                input_ids = batch["input_ids"].to(self.device)
                mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                out = self.model(input_ids, mask, labels)
                logits = out["logits"]
                
                batch_pred, batch_true = self._decode_batch(logits, labels)
                pred_tags.extend(batch_pred)
                true_tags.extend(batch_true)
                     
        metric = Metric(true_tags, pred_tags)
        p, r, f1 = metric.precision, metric.recall, metric.f1
        
        return p, r, f1, pred_tags, true_tags
