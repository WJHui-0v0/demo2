
import os
# 设置hf国内镜像加速下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download


def download_model(repo_id: str, local_folder: str):
    # autodl‑tmp永久数据盘目录，关机释放文件不会删除
    os.makedirs(local_folder, exist_ok=True)

    # NER训练必备5个pytorch文件，不下载tf冗余权重
    file_list = [
        "config.json",
        "vocab.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "pytorch_model.bin"
    ]

    for filename in file_list:
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_folder,
            force_download=False   # False开启断点续传，中断后重跑可继续下
        )
    print(f"✅ {repo_id} 下载完成！存放路径：{local_folder}")


if __name__ == "__main__":
    # bert‑base‑chinese
    download_model(
        repo_id="bert-base-chinese",
        local_folder="/root/autodl-tmp/pretrain/bert-base-chinese"
    )

    # hfl/chinese‑bert‑wwm
    download_model(
        repo_id="hfl/chinese-bert-wwm",
        local_folder="/root/autodl-tmp/pretrain/hfl-chinese-bert-wwm"
    )
