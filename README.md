
# Demo 2：中文命名实体识别（NER）

本项目基于 **BERT / BERT-WWM**预训练模型实现中文命名实体识别（NER），支持两个经典数据集：
- **MSRA** 数据集（地名、人名、组织机构）
- **Weibo** 微博数据集（细粒度人物、地名、组织）

支持四种模型组合：
1. MSRA + BERT
2. MSRA + BERT-WWM
3. Weibo + BERT
4. Weibo + BERT-WWM

##  一、项目结构

```text
demo2/
├── main.py                     # 训练入口
├── predict.py                  # 推理入口
├── dataset.py                  # 数据集和 DataCollator
├── model.py                    # BERTNER 模型
├── trainer.py                  # 训练、验证、测试循环
├── metric.py                   # 手写实体级指标
├── utils.py                    # 标签映射、随机种子、配置读取等工具
├── configs/
│   ├── labels.json             # 自动生成数据集标签映射表
│   ├── msra_bert.json          # MSRA + BERT 配置
│   ├── msra_wwm.json           # MSRA + WWM‑BERT 配置
│   ├── weibo_bert.json         # Weibo + BERT 配置
│   └── weibo_wwm.json          # Weibo + WWM‑BERT 配置
├── data/
│   ├── weibo/
│   └── msra/
├── checkpoints/                # 最优模型保存目录
└── logs/                       # SwanLab 日志目录

```

## 二、 环境安装

本项目使用的主要环境版本如下。

```text
torch==2.8.0
transformers==5.14.1
scikit-learn==1.9.0
tqdm==4.66.2
numpy==2.3.2
matplotlib==3.10.5
seaborn==0.8.1
swanlab==0.10.0
tokenizers==0.22.2
```

## 三、 数据格式

数据采用 CoNLL 风格，每行一个 token 和一个标签，句子之间用空行分隔：

```text
我 O
患 O
有 O
肺 B-LOC
炎 I-LOC

北 B-LOC.NAM
京 I-LOC.NAM
市 I-LOC.NAM
```

标签含义：

| 标签 | 含义 |
|---|---|
| `O` | 不属于任何实体 |
| `B-Type` | 一个 `Type` 实体的开始位置 |
| `I-Type` | 一个 `Type` 实体的内部位置 |

本项目使用的两个数据集标签体系不同：

| 数据集  | 标签示例 | 标签数量 | 说明 |
|---|---|---:|---|
| MSRA | `O`、`B-LOC`、`I-LOC` | 7 | 粗粒度实体类型，通常包括 `PER`、`ORG`、`LOC` 的 BIO 标签和 `O` |
| Weibo | `O`、`B-LOC.NAM`、`I-LOC.NAM` | 17 | 细粒度实体类型，类型名包含点号，例如 `LOC.NAM`、`LOC.NOM` |

标签数量由数据集实际标签文件决定。`build_label_file()` 会分别根据 `dataset_name` 将标签写入 `configs/labels.json`，训练时再由 `get_label_map(dataset_name)` 读取对应映射。

数据集格式为：

```text
data/<dataset>/train.txt
data/<dataset>/dev.txt
data/<dataset>/test.txt
```


## 四、 使用说明
本项目通过 argparse 模块实现命令行参数解析，运行不同实验
修改或指定不同的配置文件即可快速切换数据集和模型：
- 训练

```text
python main.py --config configs/msra_bert.json
```
- 推理

```text
python predict.py --config configs/weibo_wwm.json   

python predict.py --config configs/weibo_wwm.json --text "张三喜欢大连的李四"

# 单句推理 + 显示标签
python predict.py --config configs/weibo_wwm.json --text "马云去杭州开会" --show-tags
```

## 五、 实验结果与分析
- 超参数 ：batch_size=16, learning_rate=2e-5, epochs=8, max_len=128, seed=42

### （1）4组对比实验测试集指标
| 模型配置 | 精确率(P) | 召回率(R) | F1‑Score |
|:---|:---:|:---:|:---:|
| MSRA + BERT‑base‑chinese | 0.8649 | 0.8731 | 0.8690 |
| MSRA + BERT‑WWM | 0.8735 | 0.8874 | 0.8804 |
| Weibo + BERT‑base‑chinese | 0.5428 | 0.6311 | 0.5836 |
| Weibo + BERT‑WWM | 0.5609 | 0.6262 | 0.5917 |

- swanlab实验日志：https://swanlab.cn/@displan0v0/demo2/overview

### （2）结果分析
1. **数据集差异**

MSRA数据集表现远好于Weibo数据集。MSRA为新闻领域规范书面文本，人名、地名、组织机构边界清晰、句式规整；Weibo数据集源自社交媒体口语化文本，存在网络用语、非正式表达、实体边界模糊、细粒度标签（`PER.NAM`/`PER.TIT`等），识别难度显著更高，最终F‑1仅在0.58‑0.59区间

2. **WWM预训练权重带来稳定正向增益**

在两个数据集上，BERT‑WWM（全词掩码）对比原版BERT均取得了F1指标提升：
- MSRA：F1由 0.8690 → 0.8804，提升 1.14%
- Weibo：F1由 0.5836 → 0.5917，提升 0.81%

验证了**全词掩码预训练任务能够更好学习中文词语语义信息，对中文命名实体识别任务有收益**。

3. **召回率普遍高于精确率**

两组Weibo实验召回率明显高于精确率，说明模型偏向于更多地预测实体，出现一部分**实体误检（负样本被预测成实体）**，导致精确率被拉低。



