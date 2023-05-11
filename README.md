# BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability 
BiLLa是开源的推理能力增强的中英双语LLaMA模型。模型的主要特性有：
- 较大提升LLaMA的中文理解能力，并尽可能减少对原始LLaMA英文能力的损伤；
- 训练过程增加较多的任务型数据，利用ChatGPT生成解析，强化模型理解任务求解逻辑；
- 全量参数更新，追求更好的生成效果。

因个人精力有限，我未能将BiLLa与当前主流的开源大模型进行充分的对比评测。以下是经过有限的评测分析得出的结论：
- BiLLa-7B-LLM 中英语言建模能力显著优于 [Chinese-LLaMA-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)；
- BiLLa-7B-SFT 中文推理能力显著优于 [BELLE-LLaMA-Ext-7B](https://github.com/LianjiaTech/BELLE) 等模型；
- 由GPT4打分，BiLLa-7B-SFT 在英文指令上得分显著高于 [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)，中文得分持平，但解题与代码得分更高。 

欢迎开发者们使用BiLLa，更感谢开发者们协助全面评估BiLLa的各项能力！


## 模型简介

该模型以原始LLaMa模型为基础，进行了如下三个阶段的训练。
- 第一阶段：扩充中文词表，使用中文预训练预料[Wudao](https://www.sciencedirect.com/science/article/pii/S2666651021000152)、英文预训练预料[PILE](https://arxiv.org/abs/2101.00027)、翻译预料[WMT](https://www.statmt.org/wmt22/translation-task.html)的中英数据进行二次预训练。
- 第二阶段：训练数据在第一阶段基础上增加任务型数据，训练过程中两部分数据保持1:1的比例混合。任务型数据均为NLP各任务的主流开源数据，包含有数学解题、阅读理解、开放域问答、摘要、代码生成等，利用ChatGPT API为数据标签生成解析，用于训练提升模型对任务求解逻辑的理解。
- 第三阶段：保留第二阶段任务型数据，并转化为对话格式，增加其他指令数据（如[Dolly 2.0](https://github.com/databrickslabs/dolly)、[Alpaca GPT4](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM)、[COIG](https://huggingface.co/datasets/BAAI/COIG)等），进行对齐阶段的微调。

借鉴[BELLE之前的工作](https://arxiv.org/abs/2304.07854)，三阶段的训练均为全量参数的更新，未使用LoRA。

## 模型下载与使用

本项目开源的模型包含：
- 第二阶段训练的语言模型 [BiLLa-7B-LLM](https://huggingface.co/Neutralzz/BiLLa-7B-LLM) 
- 第三阶段指令微调后的模型 [BiLLa-7B-SFT](https://huggingface.co/Neutralzz/BiLLa-7B-SFT)

<b>注意</b>：因为LLaMA的License限制，本项目开放的模型权重并不能直接使用。开放的模型权重中`word embedding`的权重为训练后模型的权重和原始LLaMA权重的和，从而保证拥有LLaMA原始模型授权的开发者可以将本项目发布的模型转化成可以使用的格式。

拥有LLaMA原始模型的开发者可以通过`embedding_convert.py`完成BiLLa模型权重的还原，以下为示例：
```shell
python3 embedding_convert.py \
    --model_dir /path_to_BiLLa/BiLLa-7B-SFT \
    --meta_llama_pth_file /path_to_LLaMA/llama-7b/consolidated.00.pth
```

BiLLa-7B-SFT模型的使用可参考`eval_codes/get_model_answer.py`。

BiLLa-7B-SFT的模型输入可利用`eval_codes/conversation.py`构造，也可按以下格式自行构造（注意`Assistant:`后必须有一个空格）：
```
Human: [Your question]
Assistant: 
```

## 模型评测

### 语言建模能力
本项目在Conditional Generation任务上评估模型的语言建模能力，在纯英文和中英混合的测试集上计算模型的Perplexity指标。

纯英文测试集为1000条PILE语料，中英混合测试集为1000条PILE加1000条WuDao语料，测试集数据均未参与BiLLa的训练，评测指标如下：
| ​                          | ​中英混合 | 英文​ |
|----------------------------|:-------:|:--------:|
| Meta LLaMA 7B  | | **5.11** |
| [Chinese-LLaMA-7B](https://github.com/ymcui/Chinese-LLaMA-Alpaca)​         	| 14.63​ | 8.89​ | 
| BiLLa-7B-LLM              | **6.92** | 5.47 |

### GPT4打分
本项目将BiLLa-7B-SFT和[ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)的模型结果放在一起，由GPT4对比两模型结果进行打分。评测代码见本项目`eval_codes/`目录，该代码基本复用了[FastChat的评测代码](https://github.com/lm-sys/FastChat/tree/main/fastchat/eval)。

英文评测数据来源于[FastChat的80条问题](https://github.com/lm-sys/FastChat/blob/main/fastchat/eval/table/question.jsonl)，中文评测数据来源于[BELLE的1000条问题](https://github.com/LianjiaTech/BELLE/blob/main/eval/eval_set.json)，评测指标如下：

<p align="center">
  <img src="./images/gpt4_scores.jpg" alt="LLM-gpt4-eval" width="800">
</p>

#### 英文评测指标

| 问题类型 | 数量 | BiLLa-7B-SFT | ChatGLM-6B | 
| - | :-: | :-: | :-: |
| generic | 10 | **8.80** | 6.35|
| knowledge | 10 | **8.30** | 6.70|
| roleplay | 10 | **8.00** | 6.90 |
| common-sense | 10 | **8.30** | 7.55|
| fermi | 10 | 5.50 | **6.20** |
| counterfactual | 10 | **7.90** | 6.70 |
| coding | 7 | **6.43** | 4.86 |
| math | 3 | **8.00** | 1.67 |
| writing | 10 | **9.00** | 6.85|
| **Micro均分** |  | **7.84** | 6.39 |
| **Macro均分** |  | **7.80** | 5.97 |

#### 中文评测指标

| 问题类型 | 数量 | BiLLa-7B-SFT | ChatGLM-6B | 
| - | :-: | :-: | :-: |
| coding | 38 | **6.95** | 6.33|
| open qa | 285 | 7.42 | **7.49**|
| brainstorming | 179 | 7.84 | **8.27**|
| classification | 65 | **7.43** | 7.32|
| math | 75 | **7.01** | 4.56|
| generation | 98 | 7.88 | **8.4**|
| summarization | 40 | 7.72 | **8.78**|
| rewrite | 131 | 7.91 | **7.96**|
| closed qa | 52 | 7.54 | **8.19**|
| extract | 37 | 7.46 | **8.43**|
| **Micro均分** |  | 7.58 | **7.63** |
| **Macro均分** |  | 7.52 | **7.57** |


### 中文复杂推理

这里非常感谢学弟锴锴（[PandaLLM](https://github.com/dandelionsllm/pandallm)作者）协助完成复杂推理的评测工作！

评测集采用[LogiQA-v2](https://github.com/csitfun/LogiQA2.0)与[C3](https://dataset.org/c3/)，两数据集均未参与BiLLa的训练，评测指标如下：

| ​                          | LogiQA-v2​ |  C3-d​ |  C3-m​ |
|----------------------------|:--------:|:-------:|:--------:|
| [BELLE-LLaMA-Ext-7B](https://github.com/LianjiaTech/BELLE)​         	| 26.41​ | 29.52​ | ​28.87​ |
| [Linly-Chinese-LLaMA-7b-hf](https://github.com/CVI-SZU/Linly)​ | 25.91​ | 32.28​ | 34.52​ |
| BiLLa-7B-SFT              | **32.74** | **56.08** | **64.69** |


## 生成示例

<details>

---
Human: 女朋友对男朋友说，我都和你说了200次了，你也不长记性，又乱买东西。请问女生对男朋友说了多少次？<br/> Assistant: 我不知道具体的数量，但是我可以告诉你，当一个人多次向另一个人表达相同的想法或建议时，他们可能会感到沮丧或失望。如果你希望能够与你的女朋友建立良好的沟通，那么最好的方法可能是倾听她的想法和感受，并尽力理解她的立场。这样，你们就可以一起合作，解决问题，并且在未来更好地沟通。<br/>

---
Human: 鸡兔同笼，头共有10，足共有28，请问鸡和兔各多少只？<br/>Assistant: 设鸡的数量为x，兔的数量为y，则有以下两个方程式：<br/><br/>x + y = 10 （因为总数为10）<br/>2x + 4y = 28 （因为每只鸡有2只脚，每只兔有4只脚）<br/><br/>将第一个方程式改写为x = 10 - y，代入第二个方程式中：<br/><br/>2(10-y) + 4y = 28 <br/><br/>化简得 <br/><br/>-2y + 20 = 28 <br/><br/>-2y = -8 <br/><br/>y = 4<br/><br/>将y的值代入x = 10-y，得到<br/><br/>x = 6<br/><br/>所以鸡的数量为6只，兔的数量为4只。<br/><br/>答案为6、4。<br/> 

---
Human: 用Python写一个冒泡排序算法<br/>Assistant: 这是一个简单的冒泡排序算法的Python实现:<br/>
```python
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                
arr = [3, 5, 2, 8, 1, 6]

bubble_sort(arr)
print("排序后的数组:")
for i in range(len(arr)):
    print("%d" %arr[i])
```
输出结果为:
```
排序后的数组:
1
2
3
5
6
8
```
上面的代码中，我们定义了一个名为 `bubble_sort` 的函数，它接受一个整数数组作为输入。函数中，我们首先定义了一个 `n` 变量，用于表示数组的长度。<br/>然后我们使用一个嵌套循环来遍历数组中的每个元素。在每一次循环中，我们将当前元素与下一个元素进行比较，如果当前元素大于下一个元素，则交换这两个元素的位置。最后，我们返回排序后的数组。<br/>在主程序中，我们定义了一个包含若干个整数的数组，然后调用 `bubble_sort` 函数来对这个数组进行排序。最后，我们将排序后的数组打印出来。<br/>

</details>



## 模型局限性
当前BiLLa模型未经RLHF训练，泛化性有待观望。

BiLLa训练过程中使用了较多的任务型数据，建议减少常识类的、实时类的提问。

BiLLa训练数据中包含了多轮对话摘要数据，但未直接包含多轮对话的生成数据，因此模型多轮对话能力可能较差。

## 使用限制
本项目相关资源仅供学术研究，不得用于商业用途。BiLLa生成的内容具有随机性，因而其可能产生有害内容。本项目不承担任何关于模型输出内容的法律责任。

## 引用
如果BiLLa对你的工作有帮助，欢迎引用！
```
@misc{BiLLA,
  author = {Zhongli Li},
  title = {BiLLa: A Bilingual LLaMA with Enhanced Reasoning Ability},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Neutralzz/BiLLa}},
}
```

