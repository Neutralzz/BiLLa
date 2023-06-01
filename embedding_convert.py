import os
import torch
import argparse
import math
import time

EMBED_SUM = -3756.942626953125
EPS = 1e-4

def main(args):
    model_state_dict = torch.load(os.path.join(args.model_dir, "pytorch_model-33-of-33.bin"))
    meta_llama_state_dict = torch.load(args.meta_llama_pth_file)

    embed_weight_1 = model_state_dict["model.embed_tokens.weight"].float()

    if "model.embed_tokens.weight" in meta_llama_state_dict:
        embed_weight_2 = meta_llama_state_dict["model.embed_tokens.weight"].float()
    elif "tok_embeddings.weight" in meta_llama_state_dict:
        embed_weight_2 = meta_llama_state_dict["tok_embeddings.weight"].float()
    else:
        raise ValueError(f"The weights of word embedding are not in the {meta_llama_pth_file}")

    if args.method == "decrypt":
        embed_weight_1[:embed_weight_2.shape[0], ] -= embed_weight_2
        # The following `assert` only works for BiLLa-7B-SFT
        # tmp = embed_weight_1.sum().item()
        # print(tmp)
        # assert math.fabs(tmp - EMBED_SUM) < EPS, f"The sum of weights ({tmp}) is wrong."
        embed_weight_1 = embed_weight_1.half()
    else:
        embed_weight_1[:embed_weight_2.shape[0], ] += embed_weight_2
    
    os.rename(os.path.join(args.model_dir, "pytorch_model-33-of-33.bin"), 
                os.path.join(args.model_dir, f"pytorch_model-33-of-33.bin.backup-{int(time.time())}"))

    model_state_dict["model.embed_tokens.weight"] = embed_weight_1
    torch.save(model_state_dict, os.path.join(args.model_dir, "pytorch_model-33-of-33.bin"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Encode or decode")
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--meta_llama_pth_file", type=str)
    parser.add_argument("--method", choices=["encrypt", "decrypt"], default="decrypt")
    args = parser.parse_args()

    main(args)
