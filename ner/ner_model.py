"""
script to run inference for Nested NER tasks.
"""
import sys
import os
import torch.nn as nn
import torch.nn.functional as F

kmnlp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
sys.path.append(kmnlp_dir)

from kmnlp.utils.tokenizers import *
from kmnlp.utils.constants import *
from kmnlp.utils.dataparallel import *

from kmnlp.layers import *
from kmnlp.encoders import *


class RegionCLF(nn.Module):
    """
    used for identifying which entity region belongs to
    """

    def __init__(self, hidden_size, n_classes):
        super(RegionCLF, self).__init__()
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


class NestedNerTagger(nn.Module):
    def __init__(self, args):
        super(NestedNerTagger, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)

        self.num_bd_label = 5
        self.sentence_output_layer = nn.Linear(args.hidden_size, self.num_bd_label)

        self.bd_ids_dict = args.bd_ids_dict

        self.seq_length = args.seq_length
        self.max_region_length = args.max_region_length

        self.entity_dict = args.entity_dict
        self.label_to_entity = args.label_to_entity

        self.region_clf = RegionCLF(
            hidden_size=args.hidden_size,
            n_classes=len(self.entity_dict),
        )
        self.criterion = nn.CrossEntropyLoss()
        self.gamma = 0.3

    def forward(self, src_batch, bd_tgt_batch, seg_batch, region_tgt_batch):
        """
        Args:
            src_batch: [batch_size x seq_length]
            bd_tgt_batch: [batch_size x seq_length]
            seg_batch: [batch_size x seq_length]
            region_tgt_batch: [batch_size]
        Returns:
            loss: boundary loss + region loss
            pred: boundary pred, region pred
        """
        # Embedding.
        emb = self.embedding(src_batch, seg_batch)
        # Encoder.
        word_repr_batch = self.encoder(emb, seg_batch)

        # task1: predict the boundary
        # sentence_output for  B I O M S predicting
        # batch_size, seq_len, 5
        sentence_output = self.sentence_output_layer(word_repr_batch)

        # task 2: entity type predict
        if bd_tgt_batch is not None:
            region_emb = list()

            # ground truth: region label
            truth_regions = list()
            for word_repr, region_tgt in zip(word_repr_batch, region_tgt_batch):
                for region_pos, region_label in region_tgt.items():
                    region_emb.append(word_repr[region_pos[0]:region_pos[1] + 1])
                    truth_regions.append(region_label)

            region_outputs = self.region_clf(region_emb)

            # calculate the loss of region entity
            truth_regions = torch.tensor(truth_regions).to(region_outputs.device)

            entity_detection_loss = self.criterion(region_outputs, truth_regions)

            # calculate the loss for boundary
            bd_tgt = bd_tgt_batch.contiguous().view(-1, 1)

            one_hot = torch.zeros(bd_tgt.size(0), self.num_bd_label). \
                to(torch.device(bd_tgt.device)). \
                scatter_(1, bd_tgt, 1.0)

            sentence_output = sentence_output.contiguous().view(-1, self.num_bd_label)
            numerator = -torch.sum(nn.LogSoftmax(dim=-1)(sentence_output) * one_hot, 1)

            tgt_mask = seg_batch.contiguous().view(-1).float()
            numerator = torch.sum(tgt_mask * numerator)
            denominator = torch.sum(tgt_mask) + 1e-6
            boundary_detection_loss = numerator / denominator

            # sentence_output = sentence_output.transpose(1, 2)
            # loss = self.criterion(sentence_output, bd_tgt)
            loss = self.gamma * entity_detection_loss + (1 - self.gamma) * boundary_detection_loss

            return loss, None
        else:
            bd_tgt_batch = torch.argmax(sentence_output, dim=2)

            seq_len_batch = torch.sum(seg_batch, dim=1)

            # get all compose of region
            pred_regions = list()  # token embeddings refer to region
            for word_repr, bd_tgt, seq_len in zip(word_repr_batch, bd_tgt_batch, seq_len_batch):
                seq_len = seq_len.item()

                for start, bd_head in enumerate(bd_tgt[:seq_len]):
                    if bd_head == 1:
                        region_end_pos = self.max_region_length + start + 1
                        region_end_pos = min(region_end_pos, seq_len)

                        for end in range(start + 1, region_end_pos):
                            if bd_tgt[end] == 2:
                                pred_regions.append(word_repr[start:end + 1])

                    elif bd_head == 3:
                        pred_regions.append(word_repr[start:start + 1])

            # pred the entity type for each region
            pred_region_labels = torch.tensor([])
            if len(pred_regions) > 0:
                region_outputs = self.region_clf(pred_regions)
                pred_region_labels = torch.argmax(region_outputs, dim=1).view(-1)

            return None, (bd_tgt_batch, pred_region_labels)


def generate_pred_region_label(args, bd_tgt_batch, region_label, seg_batch, neg_label=True):
    seq_len_batch = torch.sum(seg_batch, dim=1)

    region_tgt_batch = list()
    region_label_idx = 0
    for bd_tgt, seq_len in zip(bd_tgt_batch, seq_len_batch):
        seq_len = seq_len.item()
        region_dict = dict()
        for start, sent_label in enumerate(bd_tgt[:seq_len]):
            if sent_label == 1:
                region_end_pos = args.max_region_length + start + 1
                region_end_pos = min(region_end_pos, seq_len)
                for end in range(start + 1, region_end_pos):
                    if bd_tgt[end] == 2:
                        if region_label[region_label_idx] != 0 or neg_label:
                            region_dict[(start, end)] = region_label[region_label_idx].item()
                        region_label_idx += 1

            elif sent_label == 3:
                if region_label[region_label_idx].item() != 0 or neg_label:
                    region_dict[(start, start)] = region_label[region_label_idx].item()
                region_label_idx += 1

        region_tgt_batch.append(region_dict)

    return region_tgt_batch


def generate_region_label_from_ground_truth(args, tgt_batch, seg_batch):
    """
    generate the region and corresponding entity type
    Args:
        args: global parameters
        tgt_batch: [batch_size * seq_len]
        seg_batch: [batch_size * seq_len]
    Returns:
        bd_tgt_batch: [batch_size * seq_len]
        region_tgt_batch [batch_size * num_region]
    """
    region_tgt_batch = list()
    seq_len_batch = seg_batch.sum(dim=1)

    bd_tgt_batch = torch.tensor([args.bd_ids_dict[elem.item()] for row in tgt_batch for elem in row]).reshape(
        -1, args.seq_length)

    for bd_tgt, tgt, seq_len in zip(bd_tgt_batch, tgt_batch, seq_len_batch):
        seq_len = seq_len.item()
        region_dict = dict()
        for start, sent_label in enumerate(bd_tgt[:seq_len]):
            if sent_label == 1:
                region_end_pos = args.max_region_length + start + 1
                region_end_pos = min(region_end_pos, seq_len)
                for end in range(start, region_end_pos):
                    if bd_tgt[end] == 2:
                        label_list = tgt[start:end + 1]

                        entity_label_list = [args.label_to_entity[elem.item()] for elem in label_list]

                        # check whether any E in the middle of the span
                        bd_label_list = bd_tgt[start:end + 1]

                        # if number of entities larger than 2 that means cross the entity
                        if len(set(entity_label_list)) == 1 and (2 not in bd_label_list[:-1]):
                            entity_label = entity_label_list[0]
                        else:
                            entity_label = 0
                        region_dict[(start, end)] = entity_label

            elif sent_label == 3:
                region_dict[(start, start)] = args.label_to_entity[tgt[start].item()]
        region_tgt_batch.append(region_dict)

    return bd_tgt_batch, region_tgt_batch


class NerTagger(nn.Module):
    def __init__(self, args):
        super(NerTagger, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.output_layer = nn.Linear(args.hidden_size, self.labels_num)
        self.crf_target = args.crf_target
        if args.crf_target:
            from torchcrf import CRF
            self.crf = CRF(self.labels_num, batch_first=True)
            self.seq_length = args.seq_length

    def forward(self, src, tgt, seg):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size x seq_length]
            seg: [batch_size x seq_length]
        Returns:
            loss: Sequence labeling loss.
            logits: Output logits.
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)

        # Target.
        logits = self.output_layer(output)
        if self.crf_target:
            tgt_mask = seg.type(torch.uint8)
            pred = self.crf.decode(logits, mask=tgt_mask)
            for j in range(len(pred)):
                while len(pred[j]) < self.seq_length:
                    pred[j].append(self.labels_num - 1)
            pred = torch.tensor(pred).contiguous().view(-1)
            if tgt is not None:
                loss = -self.crf(F.log_softmax(logits, 2), tgt, mask=tgt_mask, reduction='mean')
                return loss, pred
            else:
                return None, pred
        else:
            tgt_mask = seg.contiguous().view(-1).float()
            logits = logits.contiguous().view(-1, self.labels_num)
            pred = logits.argmax(dim=-1)
            if tgt is not None:
                tgt = tgt.contiguous().view(-1, 1)
                one_hot = torch.zeros(tgt.size(0), self.labels_num). \
                    to(torch.device(tgt.device)). \
                    scatter_(1, tgt, 1.0)
                numerator = -torch.sum(nn.LogSoftmax(dim=-1)(logits) * one_hot, 1)
                numerator = torch.sum(tgt_mask * numerator)
                denominator = torch.sum(tgt_mask) + 1e-6
                loss = numerator / denominator
                return loss, pred
            else:
                return None, pred
