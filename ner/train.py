import os
import sys
import argparse

"""from kmnlp.layers import *
from kmnlp.encoders import *
from kmnlp.utils.config import load_hyperparam"""

import tempfile
from typing import Dict, Iterable, List, Tuple

import allennlp
import torch
import torch.nn.functional as F
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import TextField
from allennlp.data.samplers import BucketBatchSampler
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer, PretrainedTransformerTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util
from allennlp.training.trainer import Trainer
from allennlp.training.gradient_descent_trainer import GradientDescentTrainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy

from IPython import embed


class NestedNERTsvReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            bd_indexers: Dict[str, TokenIndexer] = None,
            entity_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer(namespace="text")}
        self.bd_indexers = bd_indexers or {"tokens": SingleIdTokenIndexer(namespace="bd")}
        self.entity_indexers = entity_indexers or {"tokens": SingleIdTokenIndexer(namespace="entity")}
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue

                try:
                    text, bd, entity = line.split('\t')
                except Exception as ex:
                    print(line.split('\t'))
                    print(ex)

                text_tokens = self.tokenizer.tokenize(text)
                bd_tokens = self.tokenizer.tokenize(bd)
                entity_tokens = self.tokenizer.tokenize(entity)

                if self.max_tokens:
                    text_tokens = text_tokens[:self.max_tokens]

                text_field = TextField(text_tokens, self.token_indexers)
                bd_label_field = TextField(bd_tokens, self.bd_indexers)
                entity_label_field = TextField(entity_tokens, self.entity_indexers)

                yield Instance(
                    {"text": text_field, "bd_label": bd_label_field, "entity_label": entity_label_field})


class NestedNERClassifier(Model):
    def __init__(
            self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedding = embedder
        self.encoder = encoder

    def forward(
            self, text: TextFieldTensors, bd_label: torch.Tensor, entity_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("In model.forward(); printing here just because binder is so slow")
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedding(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = F.softmax(logits, dim=-1)
        # Shape: (1,)
        loss = F.cross_entropy(logits, bd_label)
        return {"loss": loss, "probs": probs}

    @staticmethod
    def generate_entity_embedding(sentence_label) -> List[Tuple[int, int]]:
        # generate region label
        label_region = []
        for start_idx, head in enumerate(sentence_label):
            if head == "S" or head == "B":
                # single entity
                if head == "S":
                    label_region.append((start_idx, start_idx))

                # other entity
                for end_idx, tail in enumerate(sentence_label[start_idx + 1:]):
                    if tail == "S" or tail == "E":
                        tail_idx = start_idx + 1 + end_idx
                        label_region.append((start_idx, tail_idx))

                    elif tail == "O":
                        break
        return label_region


def build_dataset_reader() -> DatasetReader:
    return NestedNERTsvReader()


def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    training_data = list(reader.read("../datasets/CMeEE/train.tsv"))
    validation_data = list(reader.read("../datasets/CMeEE/dev.tsv"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


if __name__ == '__main__':
    file_path = "../datasets/CMeEE/dev.tsv"

    text_vocab_path = "../models/zh_vocab.txt"

    bd_label_vocab_path = "../models/bd_label.txt"

    entity_label_vocab_path = "../models/entity_label.txt"

    white_tokenizer = WhitespaceTokenizer()

    sentence = "金 域 量 言 , hello !"
    print(sentence)

    tokenized_sentence = white_tokenizer.tokenize(sentence)

    print(tokenized_sentence)

    # data reader
    reader = NestedNERTsvReader(max_tokens=128)

    data_loader = MultiProcessDataLoader(reader, file_path,
                                         batch_sampler=BucketBatchSampler(batch_size=4, sorting_keys=["text"]))

    vocab = Vocabulary(oov_token='[UNK]', padding_token='[PAD]')

    vocab.set_from_file(text_vocab_path, is_padded=False, namespace="text")
    vocab.set_from_file(bd_label_vocab_path, is_padded=False, namespace="bd")
    vocab.set_from_file(entity_label_vocab_path, is_padded=False, namespace="entity")

    data_loader.index_with(vocab)

    tokens = white_tokenizer.tokenize(sentence)

    text_field = TextField(tokens, {"tokens":SingleIdTokenIndexer(namespace="text")})
    text_field.index(vocab)

    padding_lengths = text_field.get_padding_lengths()

    tensor_dict = text_field.as_tensor(padding_lengths)
    print(tensor_dict)



"""    # parameter loading for model initialization
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config_path", default="models/kmbert/base_config.json", type=str,
                        help="Path of the config file.")

    parser.add_argument("--embedding", choices=["word", "word_pos", "word_pos_seg", "word_sinusoidalpos"],
                        default="word_pos_seg",
                        help="Emebdding type.")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Max sequence length for word embedding.")
    parser.add_argument("--relative_position_embedding", action="store_true",
                        help="Use relative position embedding.")
    parser.add_argument("--relative_attention_buckets_num", type=int, default=32,
                        help="Buckets num of relative position embedding.")
    parser.add_argument("--remove_embedding_layernorm", action="store_true",
                        help="Remove layernorm on embedding.")
    parser.add_argument("--remove_attention_scale", action="store_true",
                        help="Remove attention scale.")
    parser.add_argument("--encoder", choices=["transformer", "rnn", "lstm", "gru",
                                              "birnn", "bilstm", "bigru",
                                              "gatedcnn"],
                        default="transformer", help="Encoder type.")
    parser.add_argument("--mask", choices=["fully_visible", "causal", "causal_with_prefix"], default="fully_visible",
                        help="Mask type.")
    parser.add_argument("--layernorm_positioning", choices=["pre", "post"], default="post",
                        help="Layernorm positioning.")
    parser.add_argument("--feed_forward", choices=["dense", "gated"], default="dense",
                        help="Feed forward type, specific to transformer model.")
    parser.add_argument("--remove_transformer_bias", action="store_true",
                        help="Remove bias on transformer layers.")
    parser.add_argument("--layernorm", choices=["normal", "t5"], default="normal",
                        help="Layernorm type.")
    parser.add_argument("--bidirectional", action="store_true", help="Specific to recurrent model.")
    parser.add_argument("--factorized_embedding_parameterization", action="store_true",
                        help="Factorized embedding parameterization.")
    parser.add_argument("--parameter_sharing", action="store_true", help="Parameter sharing.")
    parser.add_argument("--has_residual_attention", action="store_true", help="Add residual attention.")
    parser.add_argument("--last4layer", action="store_true",
                        help="Using the sum of last four layers of BERT as the embeddings")

    args = parser.parse_args()

    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    args.tokenizer = BertTokenizerFast(vocab_file="models/zh_vocab.txt")

    # model building
    embedding = str2embedding[args.embedding](args, len(args.tokenizer.get_vocab()))

    encoding = str2encoder[args.encoder](args)

    model = NestedNERClassifier(embedder=embedding, encoder=encoding)
    model.load_state_dict(torch.load("models/kmbert/kmbert_base.bin", map_location=torch.device('cpu')), strict=False)
"""