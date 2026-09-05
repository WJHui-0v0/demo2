

def _split_tag(tag):
    if tag == 'O':
        return 'O', ""
    if '-' not in tag:
        raise ValueError(f"非法 BIO 标签 {tag!r}")
    pre, entity_type = tag.split("-", 1)
    if pre not in {"B", "I"} or not entity_type:
        raise ValueError(f"非法 BIO 标签 {tag!r}")
    return pre, entity_type

def extract_entities(tags): #左闭右开
    entities = []
    active_type = None
    active_start = None
    
    def close_entity(end_idx):
        nonlocal active_type, active_start
        if active_type is not None:
            entities.append((active_type, active_start, end_idx))
        active_type = None
        active_start = None
    
    for i, tag in enumerate(tags):
        pre, entity_type = _split_tag(tag)
        if pre == "O":
            close_entity(i)
            
        elif pre == "B":
            close_entity(i)
            active_type = entity_type
            active_start = i
        
        elif pre == "I":
            if active_type is None or active_type != entity_type:
                close_entity(i)
                active_type = entity_type
                active_start = i
    
    close_entity(len(tags))
    return entities    


def _valid(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"真实序列数量 ({len(y_true)}) 与预测序列数量 ({len(y_pred)}) 不一致"
        )

    for sentence_index, (true_tags, pred_tags) in enumerate(zip(y_true, y_pred)):
        if len(true_tags) != len(pred_tags):
            raise ValueError(
                f"第 {sentence_index} 条序列的 token 数不一致："
                f"true={len(true_tags)}, pred={len(pred_tags)}"
            )
            
            
def entity_counts(y_true, y_pred):
    
    _valid(y_true, y_pred)
    
    true_total = 0     #tp+fn
    pred_total = 0     #tp+fp
    true_positive = 0  #tp
    
    for true_tags, pred_tags in zip(y_true, y_pred):
        true_entities = set(extract_entities(true_tags))
        pred_entities = set(extract_entities(pred_tags))
        
        true_total += len(true_entities)
        pred_total += len(pred_entities)
        
        true_positive += len(true_entities & pred_entities)  #求交集

    return true_total, pred_total, true_positive
class Metric:
    def __init__(self, y_true, y_pred):
        self.true_total, self.pred_total, self.tp = entity_counts(y_true, y_pred)
    
    @property
    def precision(self):
        return self.tp / self.pred_total if self.pred_total > 0 else 0.0

    @property
    def recall(self):
        return self.tp / self.true_total if self.true_total > 0 else 0.0

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    
    def report(self):
        print(f"Precision: {self.precision:.4f}")
        print(f"Recall: {self.recall:.4f}")
        print(f"F1 Score: {self.f1:.4f}")