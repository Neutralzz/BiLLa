# 训练语料配置
pretrain_data_root_dir = "./pretraining_corpora"
pretrain_data_split = {
    "zh": {"weight":4, "file_start": 10017},
    "en": {"weight":4, "file_start": 10041},
    "translate": {"weight":2, "file_start": 10009},
    "zh-qa-long": {"weight":1, "file_start": 10001},
    "zh-qa-short": {"weight":1, "file_start": 10001},
    "zh-reasoning": {"weight":1, "file_start": 10001},
    "zh-summary": {"weight":1, "file_start": 10001},
    "en-hotpotqa": {"weight":1, "file_start": 10001},
    "en-reasoning": {"weight":1, "file_start": 10001},
    "en-squad2": {"weight":1, "file_start": 10001},
    "en-summary": {"weight":1, "file_start": 10001},
    "en-summary-dialogue": {"weight":1, "file_start": 10001},
    "en-wikihow": {"weight":1, "file_start": 10001},
}

# 模型初始点  第一阶段训练后的模型
init_model_path = "./pretrained_models/llama-7b-stage1"

# 超参
max_seq_length = 1024
batch_size = 20 # 单卡 batch size
learning_rate = 2e-5
max_train_steps = 20000
warmup_steps = 2000

# 日志及模型输出路径
logdir = "./model_outputs/llama_task_instruct_tuning_0420/logs"
output_dir = "./model_outputs/llama_task_instruct_tuning_0420/models"

# 中断恢复
recover_training_from_checkpoint = None
