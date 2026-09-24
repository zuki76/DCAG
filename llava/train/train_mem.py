# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

import os
import sys


METHOD_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if METHOD_ROOT not in sys.path:
    sys.path.insert(0, METHOD_ROOT)


from llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

from llava.train.train import train

if __name__ == "__main__":
    train()
