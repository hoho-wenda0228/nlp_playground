import os
import sys

from allennlp.data.samplers import BucketBatchSampler

kmnlp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(kmnlp_dir)

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
from allennlp.data.fields import TextField, SequenceLabelField, MultiLabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
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
            max_tokens: int = None,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue

                try:
                    text, bd, entity = line.split('\t')
                    bd = bd.strip().split()
                    entity = entity.strip().split()
                except Exception as ex:
                    print(line.split('\t'))
                    print(ex)

                tokens = self.tokenizer.tokenize(text)

                if self.max_tokens:
                    tokens = tokens[:self.max_tokens]

                text_field = TextField(tokens, self.token_indexers)
                bd_label_field = SequenceLabelField(bd, text_field)
                entity_label_field = MultiLabelField(entity)
                yield Instance(
                    {"text": text_field, "bd_label": bd_label_field, "entity_label_field": entity_label_field})


class NestedNERClassifier(Model):
    def __init__(
            self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    def forward(
            self, text: TextFieldTensors, bd_label: torch.Tensor, entity_label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        print("In model.forward(); printing here just because binder is so slow")
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
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
    reader = NestedNERTsvReader(max_tokens=128)
    file_path = "../datasets/CMeEE/dev.tsv"

    """vocab = Vocabulary(padding_token="[PAD]", oov_token="[UNK]")
    vocab.set_from_file("zh_vocab.txt")"""

    vocab = Vocabulary.from_instances(reader.read(file_path))

    data_loader = MultiProcessDataLoader(reader, file_path,
                                         batch_sampler=BucketBatchSampler(batch_size=4, sorting_keys=["text"]))

    data_loader.index_with(vocab)
    for batch in data_loader:
        print(batch)
