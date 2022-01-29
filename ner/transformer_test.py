from transformers import BertTokenizer, BertTokenizerFast
from transformers import AutoModel, BertModel, BertConfig, AutoConfig
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys

kmnlp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(kmnlp_dir)



tokenizer = BertTokenizerFast(vocab_file="../models/zh_vocab.txt")

sentence = "金 域量言, 你好!"
print(sentence)

tokenized_sentence = tokenizer.tokenize(sentence)

print(tokenized_sentence)

ids = tokenizer.convert_tokens_to_ids(tokenized_sentence)

bert_ids = tokenizer.encode_plus(sentence, add_special_tokens=False, return_offsets_mapping=True)

config = AutoConfig.from_pretrained("../models/kmbert/base_config.json")


