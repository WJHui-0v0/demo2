
import argparse
import os
import sys
import torch
from transformers import AutoTokenizer
from model import BERTNER
from utils import get_label_map, load_config
from metric import extract_entities


def predict(text, model, tokenizer, id2label, device, max_length=128):
    if not text or not text.strip():
        return [], []
    chars = list(text.strip())
    encoded = tokenizer(
        chars,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    model_inputs = {key: value.to(device) for key, value in encoded.items()}
    model.eval()
    
    with torch.no_grad():
        logits = model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs.get("attention_mask"),
        )
        pred_ids = torch.argmax(logits, dim=-1)[0].detach().cpu().tolist()

    word_ids = encoded.word_ids(batch_index=0)
    
    tags = []
    valid_chars = []
    previous_word_id = None
    
    for pred_id, word_id in zip(pred_ids, word_ids):
        
        if word_id is None or word_id == previous_word_id:
            continue
        if word_id >= len(chars):
            continue
        
        valid_chars.append(chars[word_id])
        tags.append(id2label[int(pred_id)])
        previous_word_id = word_id
    
    # BIO实体解码函数
    raw_entities = extract_entities(tags)
    entities = []
    for entity_type, start, end in raw_entities:
        entities.append({
            "type": entity_type,
            "text": "".join(valid_chars[start:end]),
            "start": start,
            "end": end,
        })
    return entities, list(zip(valid_chars, tags))


def parse_args():
    parser = argparse.ArgumentParser(description="单句中文 NER 推理")
    parser.add_argument("--config", type=str, required=True,
                        help="json配置文件路径")
    parser.add_argument("--text", type=str, default=None, help="待识别的中文句子")
    parser.add_argument("--show-tags", action="store_true",
                        help="打印每个字符的预测标签")
    return parser.parse_args()


def main():
    args = parse_args()
    
    cfg = load_config(args.config)
    dataset_name = cfg["dataset_name"]
    max_length = cfg["max_length"]
    model_name = cfg["model_name"]
    
    checkpoint_path = cfg["save_path"]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到训练好的模型：{checkpoint_path}")
    label2id, id2label = get_label_map(dataset_name)

    id2label = {int(k): str(v) for k, v in id2label.items()}
    num_labels = len(label2id)
    cfg["num_labels"] = num_labels
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BERTNER(cfg).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    print(f"成功加载最优模型权重: {checkpoint_path}")

    text = args.text
    if text is None:
        try:
            text = input("请输入中文句子：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n程序退出。")
            sys.exit(0)

    entities, tagged_chars = predict(
        text=text,
        model=model,
        tokenizer=tokenizer,
        id2label=id2label,
        device=device,
        max_length=max_length,
    )

    print(f"\n数据集: {dataset_name} | 预训练模型: {model_name}")
    print(f"输入文本: {text}")
    if args.show_tags:
        print("字符-标签序列: " + " ".join(f"{char}/{tag}" for char, tag in tagged_chars))
    if not entities:
        print("未识别到命名实体")
        return
    print("识别出的实体列表:")
    for entity in entities:
        print(f"- {entity['text']:15s} | 实体类型: {entity['type']:8s} | 字符区间: [{entity['start']}, {entity['end']})")


if __name__ == "__main__":
    main()
