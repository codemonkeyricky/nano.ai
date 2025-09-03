import os
import numpy as np
import torch
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer


def dump_bin(tensor, filename, transpose=False):
    arr = tensor.detach().cpu().numpy()
    if transpose:
        arr = arr.T
    arr.astype(np.float32).tofile(filename)


def save_qwen_tokenizer_bin(output_path, tokenizer):
    tokens = [token for token, idx in sorted(tokenizer.vocab.items(), key=lambda x: x[1])]
    vocab_size = len(tokens)
    with open(output_path, "wb") as f:
        f.write(vocab_size.to_bytes(4, "little"))
        for token in tokens:
            token_bytes = token.encode("utf-8")
            length = len(token_bytes)
            f.write(length.to_bytes(4, "little"))
            f.write(token_bytes)


def dump_config_bin(config, filename):
    # Select fields to dump
    fields = [
        ("vocab_size", config.vocab_size),
        ("hidden_size", config.hidden_size),
        ("num_attention_heads", config.num_attention_heads),
        ("num_hidden_layers", config.num_hidden_layers),
        ("max_position_embeddings", config.max_position_embeddings),
        ("num_key_value_heads", config.num_key_value_heads),
        ("intermediate_size", config.intermediate_size),
    ]
    with open(filename, "wb") as f:
        for _, value in fields:
            f.write(int(value).to_bytes(4, "little"))


def dump_tokenizer_bin(tokenizer, filename):
    vocab = tokenizer.get_vocab()
    tokens = [token for token, idx in sorted(vocab.items(), key=lambda x: x[1])]
    vocab_size = len(tokens)
    with open(filename, "wb") as f:
        f.write(vocab_size.to_bytes(4, "little"))
        for token in tokens:
            token = token.replace("\u0120", " ")
            token = token.replace("\u010a", "\n")
            token_bytes = token.encode("utf-8")
            length = len(token_bytes)
            f.write(length.to_bytes(4, "little"))
            f.write(token_bytes)


def main():

    # Ensure output folder is fresh
    if os.path.exists("weights"):
        shutil.rmtree("weights")

    os.makedirs("weights", exist_ok=True)

    model_name = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen3-4B-Instruct-2507")

    # Select model class based on model name
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True  # reduce memory footprint
    )

    dump_config_bin(model.config, "weights/config.bin")

    # Dump tokenizer.bin
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dump_tokenizer_bin(tokenizer, "weights/tokenizer.bin")

    embeddings = model.model.embed_tokens.weight.detach().cpu()
    embeddings_numpy = embeddings.view(torch.int16).numpy()
    with open("weights/embeddings.bin", "wb") as f:
        embeddings_numpy.tofile(f)

    norm = model.model.norm.weight.detach().cpu().view(torch.int16).numpy()
    with open("weights/norm.bin", "wb") as f:
        norm.tofile(f)

    # Dump weights for all layers
    for i in range(model.config.num_hidden_layers):
        # input_layernorm
        norm_weight = model.model.layers[i].input_layernorm.weight.detach().cpu().view(torch.int16).numpy()
        with open(f"weights/layer_{i}_input_layernorm.bin", "wb") as f:
            norm_weight.tofile(f)

        # post_attention_layernorm
        norm_weight = model.model.layers[i].post_attention_layernorm.weight.detach().cpu().view(torch.int16).numpy()
        with open(f"weights/layer_{i}_post_attention_layernorm.bin", "wb") as f:
            norm_weight.tofile(f)

        # mlp weights
        gate_proj = model.model.layers[i].mlp.gate_proj.weight.detach().cpu().view(torch.int16).numpy()
        with open(f"weights/layer_{i}_mlp_gate_proj.bin", "wb") as f:
            gate_proj.tofile(f)

        up_proj = model.model.layers[i].mlp.up_proj.weight.detach().cpu().view(torch.int16).numpy()
        with open(f"weights/layer_{i}_mlp_up_proj.bin", "wb") as f:
            up_proj.tofile(f)

        down_proj = model.model.layers[i].mlp.down_proj.weight.detach().cpu().view(torch.int16).numpy()
        with open(f"weights/layer_{i}_mlp_down_proj.bin", "wb") as f:
            down_proj.tofile(f)

        # self_attn weights and biases
        attn = model.model.layers[i].self_attn

        with open(f"weights/layer_{i}_q_proj_w.bin", "wb") as f:
            q = attn.q_proj.weight.detach().cpu().view(torch.int16).numpy()
            q.tofile(f)
        # with open(f"weights/layer_{i}_q_proj_b.bin", "wb") as f:
        #     q = attn.q_proj.bias.detach().cpu().view(torch.int16).numpy()
        #     q.tofile(f)
        with open(f"weights/layer_{i}_k_proj_w.bin", "wb") as f:
            k = attn.k_proj.weight.detach().cpu().view(torch.int16).numpy()
            k.tofile(f)
        # with open(f"weights/layer_{i}_k_proj_b.bin", "wb") as f:
        #     k = attn.k_proj.bias.detach().cpu().view(torch.int16).numpy()
        #     k.tofile(f)
        with open(f"weights/layer_{i}_v_proj_w.bin", "wb") as f:
            v = attn.v_proj.weight.detach().cpu().view(torch.int16).numpy()
            v.tofile(f)
        # with open(f"weights/layer_{i}_v_proj_b.bin", "wb") as f:
        #     v = attn.v_proj.bias.detach().cpu().view(torch.int16).numpy()
        #     v.tofile(f)
        with open(f"weights/layer_{i}_o_proj_w.bin", "wb") as f:
            o = attn.o_proj.weight.detach().cpu().view(torch.int16).numpy()
            o.tofile(f)

    if hasattr(model, "lm_head"):
        lm_head_weight = model.lm_head.weight.detach().cpu().view(torch.int16).numpy()
        with open("weights/lm_head.bin", "wb") as f:
            lm_head_weight.tofile(f)

    print("dump complete!")


if __name__ == "__main__":
    main()
