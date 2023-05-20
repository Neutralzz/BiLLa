# 训练语料配置
pretrain_data_root_dir = "./pretraining_corpora"
pretrain_data_split = {
    'zh': 4,
    'en': 4,
    'translate': 2
}

# 模型初始点  原始LLaMA扩充中文词表后的模型
init_model_path = "./pretrained_models/llama-7b-emb-zh"

# 超参
max_seq_length = 1024
batch_size = 20 # 单卡 batch size
learning_rate = 3e-5
max_train_steps = 100000
warmup_steps = 2000

# 日志及模型输出路径
logdir = "./model_outputs/llama_pretrain_0331/logs"
output_dir = "./model_outputs/llama_pretrain_0331/models"

# 中断恢复
recover_training_from_checkpoint = None
