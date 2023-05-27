# BiLLa 模型训练

<b> 代码仅供参考，实际执行需补全pretraining_corpora内各子集的数据和prompts </b> 

## 运行环境

- Python >= 3.8
- PyTorch >= 1.12.0
- Colossalai >= 0.2.5
- transformers >= 4.27.0
- sentencepiece >= 0.1.97

## 训练代码

| 文件 | 描述 | GPU |
| - | - | - |
| pretrain_args.py | 第一阶段预训练的参数配置文件 | |
| pretrain_main.py | 第一阶段预训练的执行文件 | 8 x V100-32G |
| task_instruct_tuning_args.py | 第二阶段混合训练的参数配置文件 | |
| task_instruct_tuning_main.py | 第二阶段混合训练的执行文件 | 8 x V100-32G |
| alignment_sft_args.py | 第三阶段指令微调的参数配置文件 | |
| alignment_sft_main.py | 第三阶段指令微调的执行文件 | 2 x A100-40G |

<b>注意点</b>
- 以上代码仅支持单节点GPU机器运行，不支持多机多卡。
- 各阶段训练代码的差异主要在内部`DataProcessor`类。
- 运行环境和数据配置好后，各阶段训练均直接通过`torchrun --standalone --nproc_per_node=${NUM_GPU} xxx_main.py`执行。
- 模型文件`llama/llama_model.py`是基于MetaAI官方代码改造的，非HF模型。如需在开源的BiLLa模型上继续训练，请参考[HF的转换代码](https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/models/llama/convert_llama_weights_to_hf.py)进行反向操作（补充：`config.json`内`max_position_embeddings`需更名为`max_sequence_length`）。
- 每阶段的训练数据都分成了不同的集合，方便在batch内进行配比。数据预先shuffle再存到文件，各集合下的文件数需大于等于GPU卡数。

## 训练数据
| 数据集 | 任务 | 阶段一/set | 阶段二/set | 阶段三/set | 备注 |
| -  | - | :-: | :-: | :-: | - |
| [WuDao](https://www.sciencedirect.com/science/article/pii/S2666651021000152) | 语言建模 | zh | zh | | |
| [Pile](https://arxiv.org/abs/2101.00027)  | 语言建模 | en | en | | 仅使用Pile-CC和Github |
| [WMT](https://www.statmt.org/wmt22/translation-task.html) | 翻译 | translate | translate | | |
| [Ape210k](https://arxiv.org/abs/2009.11506) | 数学解题 | | zh-reasoning | mwp-zh | ChatGPT辅助生成解析 |
| [Math23k](https://aclanthology.org/D17-1088/) | 数学解题 | | zh-reasoning | mwp-zh | ChatGPT辅助生成解析 |
| [CNewsSum](https://arxiv.org/abs/2110.10874) | 文生摘 + 摘生文 | | zh-summary | others | |
| [CMRC2018](https://github.com/ymcui/cmrc2018) | 阅读理解 | | zh-qa-short |  | |
| [DuReader](https://aclanthology.org/W18-2605/) | 阅读理解 | | zh-qa-short |  | |
| [WebQA (Baidu)]() | 开放域问答 | | zh-qa-long | | |
| [MathQA](https://arxiv.org/abs/1905.13319) | 数学解题 | | en-reasoning | mwp-en | ChatGPT辅助生成解析 |
| [HotpotQA](https://aclanthology.org/D18-1259/) | 多跳阅读理解 | | en-reasoning | others | ChatGPT辅助生成解析 |
| HotpotQA | 多跳阅读理解 | | en-hotpotqa |  |  |
| [SQuAD 2.0](http://arxiv.org/abs/1806.03822) | 阅读理解 | | en-squad2 | others | |
| [WikiHow](https://github.com/mahnazkoupaee/WikiHow-Dataset) | 文生摘 + 摘生文 | | en-wikihow | | |
| [CNN DM](https://github.com/abisee/cnn-dailymail) | 文生摘 + 摘生文 | | en-summary | others | |
| [SamSum](https://aclanthology.org/D19-5409/) | 对话摘要 | | en-summary-dialogue | others | |
| [MediaSum](https://arxiv.org/abs/2103.06410) | 对话摘要 | | en-summary-dialogue | others | |
| [CodeAlpaca](https://github.com/sahil280114/codealpaca) | 指令 (代码生成) | | en-reasoning | code-alpaca-en | |
| [CoT](https://github.com/google-research/FLAN)  | 指令 | | | cot-en | |
| [COIG](https://huggingface.co/datasets/BAAI/COIG) (leetcode) | 指令 (代码生成) | | zh-reasoning | leet-code-zh | |
| COIG | 指令 | | | others | 仅使用human_value对齐数据 |
| [AlpacaGPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM) | 指令 | | | others | |
| [Dolly 2.0](https://github.com/databrickslabs/dolly) | 指令 | | | others | |
| [GPT4Tools](https://github.com/StevenGrove/GPT4Tools)  | 指令 | | | others | |
| [GPTeacher](https://github.com/teknium1/GPTeacher) | 指令 | | | others | |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3)   | 指令 | | | others | |

### 第二阶段任务数据转指令数据的过程
_本部分相对复杂，故稍作简述_

任务数据在转换成模型输入时，会搭配不同的prompt。比如文件中一条内容如下：
```
{"input": "xxx", "output": "yyy", "type": "[type]"}
```
对应在`pretraining_corpora/prompts`目录下必须存在两个文件`[type].pre.prompt`和`[type].post.prompt`，保存着“前接”prompt和“后接”prompt。

当`DataProcessor`读取到该数据时，随机选择prompt与任务数据input拼接。假设选择的prompt来自`[type].pre.prompt`，prompt就在input前面，反之位于其后。

具体见代码`task_instruct_tuning_main.py`的第172-182行，配合`pretraining_corpora/zh-summary`和`pretraining_corpora/prompts`两目录下的文件更便于理解。
