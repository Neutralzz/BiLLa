import sys, os, time
from functools import partial

import random
import datasets
import torch
import torch.distributed as dist
import transformers
from transformers import set_seed

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
from colossalai.nn.optimizer.zero_optimizer import ZeroOptimizer
from colossalai.nn.lr_scheduler import CosineAnnealingWarmupLR
from colossalai.nn.parallel import GeminiDDP
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext

from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.nn.parallel.utils import get_static_torch_model

from llama import LlamaTokenizer, LlamaConfig
from llama import LLaMaForCausalLM as LlamaForCausalLM

import alignment_sft_args as args
from copy import deepcopy

def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")
set_cpu_maximum_parallelism()

def get_tflops(model_numel, batch_size, seq_len, step_time):
    return model_numel * batch_size * seq_len * 8 / 1e12 / (step_time + 1e-12)

def logw(content):
    if not isinstance(content, str):
        content = f"{content}"
    content = f'[{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}] {content}'
    print(content, file=open(args.logdir+f'/log.txt.{dist.get_rank()}', 'a', encoding='utf-8'))

import threading
from queue import Queue
class DataProcessor(threading.Thread):
    def __init__(self, data_root, data_files, tokenizer, batch_size, max_seq_length):
        super().__init__()
        self.data_root = data_root
        self.data_files = data_files
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length

        self.queue = Queue(100)

        self.raw_datasets = {
            k: {
                'file_index': 0,
                'dataset': datasets.load_dataset('json', 
                            data_files=os.path.join(self.data_root, k, v['files'][0]), split='train'),
                'sample_index': 0
            }
            for k, v in self.data_files.items()
        }

        self.running = True

    def get_sample(self, tag):
        if tag not in self.raw_datasets:
            return None
        sample_index = self.raw_datasets[tag]['sample_index']
        file_index   = self.raw_datasets[tag]['file_index']
        dataset = self.raw_datasets[tag]['dataset']
        sample = dataset[sample_index]

        if sample_index + 1 >= len(dataset):
            num_files = len(self.data_files[tag]['files'])
            file_index = (file_index + 1) % num_files

            logw(f"[data] loading {tag}: {self.data_files[tag]['files'][file_index]}")
            self.raw_datasets[tag]['dataset'] = datasets.load_dataset('json',
                            data_files=os.path.join(self.data_root, tag, self.data_files[tag]['files'][file_index]), split='train')
            self.raw_datasets[tag]['sample_index'] = 0
            self.raw_datasets[tag]['file_index'] = file_index
        else:
            self.raw_datasets[tag]['sample_index'] += 1
            if self.raw_datasets[tag]['sample_index'] % 10000 == 0:
                logw(f"[data] {tag} {self.data_files[tag]['files'][file_index]} {self.raw_datasets[tag]['sample_index']}/{len(dataset)}")

        return sample

    def make_features(self, examples):
        # print(examples)
        features = {
            "input_ids": [],
            "loss_mask": []
        }

        max_length_in_examples = 0
        for example in examples:
            prefix = "Human: "
            for key in example:
                example[key] = example[key].strip()
            assert len(example["instruction"]) + len(example["input"]) > 0 
            if len(example["instruction"]) > 0:
                prefix += example["instruction"] + "\n"
            if len(example["input"]) > 0:
                prefix += example["input"] + "\n"
            prefix += "Assistant: "

            input_ids_1 = self.tokenizer.encode(prefix, True, False)
            loss_mask_1 = [0] * len(input_ids_1)
            input_ids_2 = self.tokenizer.encode(example['output'], False, True)
            loss_mask_2 = [1] * len(input_ids_2)
            input_ids = input_ids_1 + input_ids_2
            loss_mask = loss_mask_1 + loss_mask_2
            input_ids = input_ids[:self.max_seq_length]
            loss_mask = loss_mask[:self.max_seq_length]

            features["input_ids"].append(input_ids)
            features["loss_mask"].append(loss_mask)
            max_length_in_examples = max(max_length_in_examples, len(input_ids))

        for i in range(len(examples)):
            features["input_ids"][i] = features["input_ids"][i] + [0] * (max_length_in_examples - len(features["input_ids"][i]))
            features["loss_mask"][i] = features["loss_mask"][i] + [0] * (max_length_in_examples - len(features["loss_mask"][i]))

        for key in features:
            features[key] = torch.tensor(features[key], dtype=torch.long)
        return features

    def run(self):
        tags = []
        for tag in self.data_files:
            tags += [tag] * self.data_files[tag]['weight']
        
        tid = 0
        batch = []
        while self.running and tid < len(tags):
            tag = tags[tid]
            batch.append(self.get_sample(tag))
            if len(batch) == self.batch_size:
                self.queue.put(self.make_features(batch))
                batch = []
            tid = (tid + 1) % len(tags)

def get_data_files():
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    data_root = args.pretrain_data_root_dir
    data_files = deepcopy(args.pretrain_data_split)

    for subdir in data_files:
        data_files[subdir]['files'] = []
        for filename in sorted(os.listdir(os.path.join(data_root, subdir))):
            if not filename.endswith('.jsonl'):
                continue
            file_index = int(filename.split('.jsonl')[0])
            if file_index < data_files[subdir]['file_start']:
                continue
            data_files[subdir]['files'].append(filename)
    for subdir in data_files:
        files = data_files[subdir]['files']
        data_files[subdir]['files'] = [files[i] for i in range(local_rank, len(files), world_size)]

    return data_files


def main():
    colossalai.launch_from_torch({})
    logger = get_dist_logger()
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()

    os.makedirs(args.logdir, exist_ok=True)

    is_main_process = local_rank == 0
    if is_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    config = LlamaConfig.from_pretrained(args.init_model_path, use_cache=False)
    assert args.max_seq_length <= config.max_sequence_length
    tokenizer = LlamaTokenizer(args.init_model_path)

    data_files = get_data_files()
    logw(data_files)
    data_processor = DataProcessor(args.pretrain_data_root_dir, data_files,
                     tokenizer, args.batch_size, args.max_seq_length)
    data_processor.start()

    with ColoInitContext(device=get_current_device(), dtype=torch.half):
        init_path = args.init_model_path if args.recover_training_from_checkpoint is None else args.recover_training_from_checkpoint
        model = LlamaForCausalLM.from_pretrained(init_path, config=config)
    model.gradient_checkpointing_enable()
    
    numel = sum([p.numel() for p in model.parameters()])
    logger.info(f"Num of parameters: {numel}")
    
    PLACEMENT_POLICY = 'cpu'
    model = GeminiDDP(model, device=get_current_device(), placement_policy=PLACEMENT_POLICY, pin_memory=True)

    # 冻结 ffn 参数
    # optimizing_parameters = []
    # for n, p in model.named_parameters():
    #     if 'feed_forward.w' not in n:
    #         optimizing_parameters.append(p)
    optimizing_parameters = model.parameters()
    # 配置 optimizer
    optimizer = HybridAdam(optimizing_parameters, lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    optimizer = ZeroOptimizer(optimizer, model, initial_scale=2**14, gpu_margin_mem_ratio=0.0, clipping_norm=2.0)
    # 配置 scheduler
    lr_scheduler = CosineAnnealingWarmupLR(optimizer, args.max_train_steps, args.warmup_steps, args.learning_rate * 0.1)
    logger.info(f"The model and optimizer is set completely.")

    recover_step = None
    if args.recover_training_from_checkpoint is not None:
        basename = os.path.basename(args.recover_training_from_checkpoint.rstrip('/'))
        recover_step = int(basename.split('step-')[1].split('_')[0])
        logger.info(f"Recovering from the previous training step {recover_step}.")
        if os.path.exists(f"{args.recover_training_from_checkpoint}/optim.{local_rank}.bin"):
            optim_state_dict = torch.load(f"{args.recover_training_from_checkpoint}/optim.{local_rank}.bin")
            optimizer.load_state_dict(optim_state_dict)
    # The above code is FINE !

    get_tflops_func = partial(get_tflops, numel, args.batch_size, args.max_seq_length)
    tflops_record = []
    loss_record   = []
    logging_steps = 100
    saving_steps  = 1000

    model.train()
    for step in range(args.max_train_steps):
        st_time = time.time()
        batch = data_processor.queue.get(True, 60)
        if recover_step is not None and step < recover_step:
            lr_scheduler.step()
            continue
        input_ids = batch["input_ids"].to(get_current_device())
        loss_mask = batch["loss_mask"].to(get_current_device())

        outputs = model(input_ids=input_ids, loss_mask=loss_mask, labels=input_ids)
        loss = outputs[0]

        loss_record.append(loss.item()) 
        
        optimizer.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        step_time = time.time() - st_time
        step_tflops = get_tflops_func(step_time)
        tflops_record.append(step_tflops)

        if (step + 1) % logging_steps == 0:
            avg_loss = sum(loss_record[-logging_steps:]) / logging_steps
            avg_tflops = sum(tflops_record[-logging_steps:]) / logging_steps
            logger.info(f"step {step+1}: lr {lr_scheduler.get_last_lr()[0]}  loss {avg_loss}  tflops {avg_tflops}")
            logw(f"[training] step {step+1}: lr {lr_scheduler.get_last_lr()[0]}  loss {avg_loss}  tflops {avg_tflops}")
        
        if (step + 1) % saving_steps == 0:
            model_to_save = get_static_torch_model(model).half()
            if is_main_process:
                avg_loss = sum(loss_record[-saving_steps:]) / saving_steps
                save_dir = os.path.join(args.output_dir, "ckpt_step-%d_loss-%.4f"%(step+1, avg_loss))
                os.makedirs(save_dir, exist_ok=True)
                model_to_save.config.save_pretrained(save_dir)
                torch.save(model_to_save.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
            dist.barrier()
            save_dir = None
            for subdir in os.listdir(args.output_dir):
                if subdir.startswith("ckpt_step-%d_loss-"%(step+1)):
                    save_dir = os.path.join(args.output_dir, subdir)
                    break
            if save_dir is not None:
                torch.save(optimizer.state_dict(), os.path.join(save_dir, f'optim.{local_rank}.bin'))

    torch.cuda.empty_cache()
    data_processor.running = False
    while not data_processor.queue.empty():
        data_processor.queue.get()

if __name__=='__main__':
    main()
