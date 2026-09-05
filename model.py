
import torch.nn as nn
from transformers import BertModel

class BERTNER(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_labels = cfg["num_labels"]
        
        self.bert = BertModel.from_pretrained(cfg["model_name"])
        hidden_size = self.bert.config.hidden_size
        
        self.dropout = nn.Dropout(cfg.get("dropout", 0.1))
        self.classifier = nn.Linear(hidden_size, self.num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    def forward(self, input_ids, attention_mask,labels=None):
        outputs = self.bert(input_ids, attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
        return {
            "loss": loss,
            "logits": logits
        }