# 训练语料配置
pretrain_data_root_dir = "./pretraining_corpora"
pretrain_data_split = {
    "code-alpaca-en": {"weight":1, "file_start": 10001},
    "cot-en": {"weight":2, "file_start": 10001},
    "leet-code-zh": {"weight":1, "file_start": 10001},
    "mwp-en": {"weight":1, "file_start": 10001},
    "mwp-zh": {"weight":1, "file_start": 10001},
    "others": {"weight":6, "file_start": 10001},
}

# 模型初始点  第二阶段训练后的模型
init_model_path = "./pretrained_models/llama-7b-stage2"

# 超参
max_seq_length = 1024
batch_size = 24  # 单卡 batch size
learning_rate = 2e-5
max_train_steps = 20000
warmup_steps = 2000

# 日志及模型输出路径
logdir = "./model_outputs/llama_alignment_sft_0501/logs"
output_dir = "./model_outputs/llama_alignment_sft_0501/models"

# 中断恢复
recover_training_from_checkpoint = None
