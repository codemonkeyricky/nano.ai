import os
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def dump_bin(tensor, filename, transpose=False):
    arr = tensor.detach().cpu().numpy()
    if transpose:
        arr = arr.T
    arr.astype(np.float32).tofile(filename)

def save_gpt2_tokenizer_bin(output_path, model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokens = [token for token, idx in sorted(tokenizer.encoder.items(), key=lambda x: x[1])]
    vocab_size = len(tokens)
    with open(output_path, "wb") as f:
        f.write(vocab_size.to_bytes(4, "little"))
        for token in tokens:
            decoded_token = tokenizer.decode([tokenizer.encoder[token]])
            token_bytes = decoded_token.encode("utf-8")
            length = len(token_bytes)
            f.write(length.to_bytes(4, "little"))
            f.write(token_bytes)

def dump_config_bin(config, filename):
    # Select fields to dump
    fields = [
        ("vocab_size", config.vocab_size),
        ("n_positions", config.n_positions),
        ("n_embd", config.n_embd),
        ("n_layer", config.n_layer),
        ("n_head", config.n_head),
    ]
    with open(filename, "wb") as f:
        for _, value in fields:
            f.write(int(value).to_bytes(4, "little"))

def main():
    # Possible models: gpt2, gpt2-medium, gpt2-large, gpt2-xl
    model_name = "gpt2"  # gpt2-small
    out_dir = "./gpt2_weights"
    os.makedirs(out_dir, exist_ok=True)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Dump config
    config_bin_path = os.path.join(out_dir, "config.bin")
    dump_config_bin(model.config, config_bin_path)

    # wte, wpe
    dump_bin(model.transformer.wte.weight, os.path.join(out_dir, "wte_weight.bin"))
    dump_bin(model.transformer.wpe.weight, os.path.join(out_dir, "wpe_weight.bin"))

    # transformer blocks
    for i, block in enumerate(model.transformer.h):
        # attn.c_attn
        dump_bin(block.attn.c_attn.weight, os.path.join(out_dir, f"h.{i}.attn_c_weight.bin"), transpose=True)
        if block.attn.c_attn.bias is not None:
            dump_bin(block.attn.c_attn.bias, os.path.join(out_dir, f"h.{i}.attn_c_bias.bin"))
        # ln_1
        dump_bin(block.ln_1.weight, os.path.join(out_dir, f"h.{i}.ln_1_weight.bin"))
        if block.ln_1.bias is not None:
            dump_bin(block.ln_1.bias, os.path.join(out_dir, f"h.{i}.ln_1_bias.bin"))
        # ln_2
        dump_bin(block.ln_2.weight, os.path.join(out_dir, f"h.{i}.ln_2_weight.bin"))
        if block.ln_2.bias is not None:
            dump_bin(block.ln_2.bias, os.path.join(out_dir, f"h.{i}.ln_2_bias.bin"))
        # attn.c_proj
        dump_bin(block.attn.c_proj.weight, os.path.join(out_dir, f"h.{i}.attn_c_proj_weight.bin"), transpose=True)
        if block.attn.c_proj.bias is not None:
            dump_bin(block.attn.c_proj.bias, os.path.join(out_dir, f"h.{i}.attn_c_proj_bias.bin"))
        # mlp.c_fc
        dump_bin(block.mlp.c_fc.weight, os.path.join(out_dir, f"h.{i}.mlp.c_fc_weight.bin"), transpose=True)
        if block.mlp.c_fc.bias is not None:
            dump_bin(block.mlp.c_fc.bias, os.path.join(out_dir, f"h.{i}.mlp.c_fc_bias.bin"))
        # mlp.c_proj
        dump_bin(block.mlp.c_proj.weight, os.path.join(out_dir, f"h.{i}.mlp.c_proj_weight.bin"), transpose=True)
        if block.mlp.c_proj.bias is not None:
            dump_bin(block.mlp.c_proj.bias, os.path.join(out_dir, f"h.{i}.mlp.c_proj_bias.bin"))

    # ln_f, lm_head
    dump_bin(model.transformer.ln_f.weight, os.path.join(out_dir, "ln_f_weight.bin"))
    if model.transformer.ln_f.bias is not None:
        dump_bin(model.transformer.ln_f.bias, os.path.join(out_dir, "ln_f_bias.bin"))
    dump_bin(model.lm_head.weight, os.path.join(out_dir, "lm_head_weight.bin"))

    # Dump tokenizer tokens
    tokenizer_bin_path = os.path.join(out_dir, "tokenizer.bin")
    save_gpt2_tokenizer_bin(tokenizer_bin_path, model_name)

if __name__ == "__main__":
    main()
