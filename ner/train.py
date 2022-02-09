import os
import sys
import argparse

from transformers import BertTokenizerFast

kmnlp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(kmnlp_dir)

from kmnlp.layers import *
from kmnlp.encoders import *
from kmnlp.utils.config import load_hyperparam

import tempfile
from typing import Dict, Iterable, List, Tuple

import allennlp
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import MultiProcessDataLoader
from allennlp.data.fields import TextField, MetadataField, TensorField
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
from allennlp.training.metrics import Metric, FBetaMeasure

from IPython import embed


class NestedNERTsvReader(DatasetReader):
    def __init__(
            self,
            tokenizer: Tokenizer = None,
            token_indexers: Dict[str, TokenIndexer] = None,
            boundary_indexers: Dict[str, TokenIndexer] = None,
            entity_indexers: Dict[str, TokenIndexer] = None,
            max_tokens: int = 128,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.max_tokens = max_tokens
        self.vocab = self.initial_vocab()

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for idx, line in enumerate(lines):
                if idx == 0:
                    continue
                try:
                    text, boundary, entity = line.split('\t')
                except Exception as ex:
                    print(line.split('\t'))
                    print(ex)
                    continue
                instance = self.text_to_instance(text, boundary, entity)
                if instance is not None:
                    yield instance

    def text_to_instance(self, text: str, boundary: str, entity: str) -> Instance:

        text_tokens = self.tokenizer.tokenize(text)[:self.max_tokens]
        boundary_tokens = self.tokenizer.tokenize(boundary)[:self.max_tokens]
        entity_tokens = self.tokenizer.tokenize(entity)

        text_field = TextField(text_tokens, {"text": SingleIdTokenIndexer(namespace="text")})
        boundary_field = TextField(boundary_tokens, {"boundary": SingleIdTokenIndexer(namespace="boundary")})
        entity_field = TextField(entity_tokens, {"entity": SingleIdTokenIndexer(namespace="entity")})

        indexed_text = self.get_token_index(text_tokens, "text")
        indexed_boundary = self.get_token_index(boundary_tokens, "boundary")
        indexed_entity = self.get_token_index(entity_tokens, "entity")

        text_len = len(indexed_text)
        seg = [1] * text_len
        seg.extend([0] * (self.max_tokens - text_len))

        indexed_text.extend([0] * (self.max_tokens - text_len))
        indexed_boundary.extend([0] * (self.max_tokens - text_len))

        text_tensor = torch.tensor(indexed_text, dtype=torch.long)
        boundary_tensor = torch.tensor(indexed_boundary, dtype=torch.long)
        entity_tensor = torch.tensor(indexed_entity, dtype=torch.long)
        seg_tensor = torch.tensor(seg, dtype=torch.long)

        return Instance({"text_batch": TensorField(text_tensor), "bd_label_batch": TensorField(boundary_tensor),
                         "entity_label_batch": MetadataField(entity_tensor), "seg_batch": TensorField(seg_tensor)})

    @staticmethod
    def initial_vocab():
        vocab = Vocabulary(oov_token='[UNK]', padding_token='[PAD]')

        text_vocab_path = "../models/zh_vocab.txt"
        bd_label_vocab_path = "../models/bd_label.txt"
        entity_label_vocab_path = "../models/entity_label.txt"

        vocab.set_from_file(text_vocab_path, is_padded=False, namespace="text")
        vocab.set_from_file(bd_label_vocab_path, is_padded=False, namespace="boundary")
        vocab.set_from_file(entity_label_vocab_path, is_padded=False, namespace="entity")
        return vocab

    def get_token_index(self, tokens: list, namespace: str):
        text_field = TextField(tokens, {namespace: SingleIdTokenIndexer(namespace=namespace)})
        text_field.index(self.vocab)
        return text_field._indexed_tokens[namespace]["tokens"]


class RegionCLF(nn.Module):
    """
    used for identifying which entity region belongs to
    """

    def __init__(self, hidden_size, n_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.fc = nn.Linear(self.hidden_size, n_classes)

    def forward(self, data_list):
        """
        Args:
            data_list: num_region * [num_token * hidden_size]
        Returns:
            region_repr: [num_region * num_entity_type]
        """
        region_repr_list = [hidden.mean(dim=0).view(1, -1) for hidden in data_list]  # num_region * [1 * hidden_size]
        region_repr = torch.cat(region_repr_list, dim=0)  # [num_region * hidden_size]

        return self.fc(region_repr)  # [batch_size x n_classes]


def generate_region(boundary_label: list) -> List[Tuple[int, int]]:
    region_list = []

    begin_label = 1
    middle_label = 2
    end_label = 3
    single_label = 4
    out_label = 0

    for start_idx, head in enumerate(boundary_label):
        if head == single_label or head == begin_label:
            # single entity
            if head == single_label:
                region_list.append((start_idx, start_idx))

            # other entity
            for end_idx, tail in enumerate(boundary_label[start_idx + 1:]):
                if tail == single_label or tail == end_label:
                    tail_idx = start_idx + 1 + end_idx
                    region_list.append((start_idx, tail_idx))

                elif tail == out_label:
                    break

    return region_list


class NERF1Measure:
    def __init__(self):
        self.tp = 0
        self.fn = 0
        self.fp = 0

    def __call__(
            self,
            region_tgt_batch,
            entity_tgt_batch,
            region_pred_batch,
            entity_pred_batch,
    ):
        entity_tgt_list, entity_pred_list = [], []
        region_true_count, region_pred_count = 0, 0

        for region_tgt, entity_tgt, region_pred, entity_pred in zip(region_tgt_batch, entity_tgt_batch,
                                                                    region_pred_batch, entity_pred_batch):
            # id2entity = lambda x: list(args.entity_dict.keys())[list(args.entity_dict.values()).index(x)]

            region_tgt_dict = dict(zip(region_tgt, entity_tgt.tolist()))
            region_pred_dict = dict(zip(region_pred, entity_pred.tolist()))

            for region in region_tgt_dict:
                true_label = region_tgt_dict[region]
                pred_label = region_pred_dict[region] if region in region_pred_dict else 0
                entity_tgt_list.append(true_label)
                entity_pred_list.append(pred_label)
            for region in region_pred_dict:
                if region not in region_tgt_dict:
                    entity_pred_list.append(region_pred_dict[region])
                    entity_tgt_list.append(0)

            if len(entity_tgt) > 0:
                region_true_count += (entity_tgt > 0).sum().item()

            if len(entity_pred) > 0:
                region_pred_count += (entity_pred > 0).sum().item()

        tp = 0
        for pv, tv in zip(entity_pred_list, entity_tgt_list):
            if pv == tv and pv != 0:
                tp += 1

        fp = region_pred_count - tp
        fn = region_true_count - tp

        self.tp += tp
        self.fp += fp
        self.fn += fn

    def get_metric(self, reset: bool = False):
        precision = 0 if self.tp + self.fp == 0 else self.tp / (self.tp + self.fp)
        recall = 0 if self.tp + self.fn == 0 else self.tp / (self.tp + self.fn)
        f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        print("get metric", {"precision": precision, "recall": recall, "f1": f1})
        embed()
        if reset:
            self.reset()
        return {"precision:": precision, "recall": recall, "f1": f1}

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0


class NestedNERClassifier(Model):
    def __init__(
            self, args, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedding = embedder
        self.encoder = encoder

        self.num_bd_label = 5
        self.sentence_output_layer = nn.Linear(args.hidden_size, self.num_bd_label)

        self.entity_clf = RegionCLF(
            hidden_size=args.hidden_size,
            n_classes=10,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.gamma = 0.3

        self.accuracy = NERF1Measure()

    def forward(
            self, text_batch: torch.Tensor, seg_batch: torch.Tensor, bd_label_batch: torch.Tensor = None,
            entity_label_batch: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        emb_batch = self.embedding(text_batch, seg_batch)
        # Shape: (batch_size, encoding_dim)
        word_repr_batch = self.encoder(emb_batch, seg_batch)

        # task1: predict the boundary
        # sentence_output for  B I O M S predicting
        # batch_size, seq_len, 5
        sentence_output = self.sentence_output_layer(word_repr_batch)

        if bd_label_batch is not None:
            entity_emb, flat_entity_label = [], []

            entity_logit, region_batch = self.generate_pred_region_by_boundary(word_repr_batch, bd_label_batch)

            for entity_label in entity_label_batch:
                flat_entity_label.extend(entity_label.tolist())

            # calculate the loss of region entity
            truth_regions = torch.tensor(flat_entity_label).to(entity_logit.device)

            entity_detection_loss = self.criterion(entity_logit, truth_regions)

            # calculate the loss for boundary
            bd_tgt = bd_label_batch.contiguous().view(-1, 1)

            one_hot = torch.zeros(bd_tgt.size(0), self.num_bd_label). \
                to(torch.device(bd_tgt.device)). \
                scatter_(1, bd_tgt, 1.0)

            sentence_output_flat = sentence_output.contiguous().view(-1, self.num_bd_label)
            numerator = -torch.sum(nn.LogSoftmax(dim=-1)(sentence_output_flat) * one_hot, 1)

            tgt_mask = seg_batch.contiguous().view(-1).float()
            numerator = torch.sum(tgt_mask * numerator)
            denominator = torch.sum(tgt_mask) + 1e-6
            boundary_detection_loss = numerator / denominator

            # sentence_output = sentence_output.transpose(1, 2)
            # loss = self.criterion(sentence_output, bd_label_batch,seg_batch)
            loss = self.gamma * entity_detection_loss + (1 - self.gamma) * boundary_detection_loss

            # evaluation
            bd_pred_batch = torch.argmax(sentence_output, dim=2)
            entity_pred_logit, region_pred_batch = self.generate_pred_region_by_boundary(word_repr_batch, bd_pred_batch)

            entity_pred_flat = torch.argmax(entity_pred_logit, dim=1).view(-1) \
                if entity_pred_logit is not None else torch.tensor([])

            entity_pred_batch = []
            for idx, region_pred in enumerate(region_pred_batch):
                start = 0 if idx == 0 else len(entity_pred_batch[idx - 1])
                end = len(region_pred) + start
                entity_pred_batch.append(entity_pred_flat[start:end])

            self.accuracy(region_batch, entity_label_batch, region_pred_batch, entity_pred_batch)

            return {"loss": loss}

        else:
            bd_label_batch = torch.argmax(sentence_output, dim=2)

            entity_logit, _ = self.generate_pred_region_by_boundary(word_repr_batch, bd_label_batch)

            entity_pred = torch.argmax(entity_logit, dim=1).view(-1) if entity_logit is not None else torch.tensor([])

            return {"bd_pred": bd_label_batch, "entity_pred": entity_pred}

    def generate_pred_region_by_boundary(self, word_repr_batch, bd_label_batch):
        # get all compose of region
        entity_emb = []  # token embeddings refer to region

        region_batch = []

        for word_repr, bd_label in zip(word_repr_batch, bd_label_batch):
            region = generate_region(bd_label.tolist())
            region_batch.append(region)
            for (start, end) in region:
                entity_emb.append(word_repr[start:end + 1])

        # pred the entity type for each region
        entity_logit = None
        if len(entity_emb) > 0:
            entity_logit = self.entity_clf(entity_emb)

        return entity_logit, region_batch

    def get_metrics(self, reset: bool = False) -> dict[str, dict[str, float]]:
        return {"accuracy": self.accuracy.get_metric(reset)}


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


def build_trainer(
        model: Model,
        serialization_dir: str,
        train_loader: DataLoader,
        dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
        cuda_device=-1,
    )
    return trainer


def intial_args():
    # parameter loading for model initialization
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config_path", default="../models/kmbert/base_config.json", type=str,
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
    return args


if __name__ == '__main__':
    train_file_path = "../datasets/CMeEE/train.tsv"
    dev_file_path = "../datasets/CMeEE/dev.tsv"

    text_vocab_path = "../models/zh_vocab.txt"

    bd_label_vocab_path = "../models/bd_label.txt"

    entity_label_vocab_path = "../models/entity_label.txt"

    # initial vocab
    vocab = Vocabulary(oov_token='[UNK]', padding_token='[PAD]')

    vocab.set_from_file(text_vocab_path, is_padded=False, namespace="text")
    vocab.set_from_file(bd_label_vocab_path, is_padded=False, namespace="boundary")
    vocab.set_from_file(entity_label_vocab_path, is_padded=False, namespace="entity")

    # data reader
    reader = NestedNERTsvReader(max_tokens=128)

    train_data_loader = MultiProcessDataLoader(
        reader, train_file_path, batch_sampler=BucketBatchSampler(batch_size=4, sorting_keys=["text_batch"]))

    dev_data_loader = MultiProcessDataLoader(
        reader, dev_file_path, batch_sampler=BucketBatchSampler(batch_size=4, sorting_keys=["text_batch"]))

    train_data_loader.index_with(vocab)
    dev_data_loader.index_with(vocab)

    args = intial_args()
    # Load the hyperparameters of the config file.
    args = load_hyperparam(args)

    args.tokenizer = BertTokenizerFast(vocab_file="../models/zh_vocab.txt")

    # model building
    embedding = str2embedding[args.embedding](args, len(args.tokenizer.get_vocab()))

    encoding = str2encoder[args.encoder](args)

    model = NestedNERClassifier(args, vocab, embedder=embedding, encoder=encoding)
    model.load_state_dict(torch.load("../models/kmbert/kmbert_base.bin", map_location=torch.device('cpu')),
                          strict=False)

    serialization_dir = "train_record"
    trainer = build_trainer(model, serialization_dir, train_data_loader, dev_data_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")
