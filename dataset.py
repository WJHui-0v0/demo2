import torch
from torch.utils.data import Dataset
from utils import get_label_map
from torch.nn.utils.rnn import pad_sequence


class NERDataset(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        # 对应文件路径
        path_map = {
            "train": cfg["train_path"],
            "dev": cfg["dev_path"],
            "test": cfg["test_path"]
        }
        if mode not in path_map:
            raise ValueError("mode只能填train/dev/test")
        self.data_path = path_map[mode]
         
        self.dataset_name = self.cfg["dataset_name"]
        self.max_length = self.cfg["max_length"]
 
        self.label2id, self.id2label = get_label_map(self.dataset_name)
        self.data = self.load_data()
        
    def load_data(self):
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            content = f.read()
        for block in content.strip().split('\n\n'):
            if not block:
                continue
            words, tags = [], []
            for line in block.split('\n'):
                line = line.strip()
                if not line:
                    continue
                word, tag = line.split()
                words.append(word)
                tags.append(tag if tag != '0' else 'O')
            data.append((words, tags))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        words, tags = self.data[idx]
        return words, tags

class DataCollatorForNER:
    def __init__(self, tokenizer,label2id, max_length):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __call__(self, batch):
        words_list = [item[0] for item in batch]
        tags_list = [item[1] for item in batch]

        tokenized = self.tokenizer(
            words_list,
            is_split_into_words=True,       #不要二次分词
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )

        batch_labels = []
        
        for idx in range(len(batch)):         #几个句子
            word_ids = tokenized.word_ids(batch_index=idx)   #获取每句下标
            raw_tags = tags_list[idx]
            label_id = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_id.append(-100)  # 特殊token
                elif word_idx != previous_word_idx:
                    tag = raw_tags[word_idx] if word_idx < len(raw_tags) else 'O'
                    label_id.append(self.label2id[tag])
                    previous_word_idx = word_idx
                else:  #上一个分裂的
                    label_id.append(-100)  # 对应subword的token

            batch_labels.append(torch.tensor(label_id, dtype=torch.long))
        padded_labels = pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        tokenized["labels"] = padded_labels
        return tokenized


if __name__ == "__main__":
    from config import load_config
    cfg = load_config()
    dataset = NERDataset(cfg, mode="train")
    print("数据数量:",len(dataset))
    item = dataset[0]
    print(item)
    