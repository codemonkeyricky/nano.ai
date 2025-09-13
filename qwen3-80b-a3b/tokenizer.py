from tokenizers import Tokenizer
import sys
import os

model_name = os.environ.get("QWEN_MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
tokenizer = Tokenizer.from_pretrained(model_name)

# Read input from stdin
input_text = sys.stdin.read().strip()
words = input_text.split()

prefix = [151644, 872, 198]
suffix = [151645, 198, 151644, 77091, 198]

input_ids = []
for word in words:
    input_ids.extend(tokenizer.encode(word).ids)

# Prepend prefix and append suffix
input_ids = prefix + input_ids + suffix

print(" ".join(str(x) for x in input_ids))

