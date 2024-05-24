import numpy as np
import torch
from torch.utils.data import Dataset


class CustomTrainDataset(Dataset):
    def __init__(self, pad_token_id, examples, hyps_refs=None):
        self.examples = examples
        self.hyps_refs = hyps_refs
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = np.array(example.features["input_ids"], dtype=np.int16)
        input_mask = np.array(example.features["input_mask"], dtype=np.int8)
        segment_ids = np.array(example.features["segment_ids"], dtype=np.int8)
        input_ids_for_subwords = np.array(example.features["input_ids_for_subwords"], dtype=np.int32)
        input_mask_for_subwords = np.array(example.features["input_mask_for_subwords"], dtype=np.int8)
        segment_ids_for_subwords = np.array(example.features["segment_ids_for_subwords"], dtype=np.int8)
        character_pos_to_subword_pos = np.array(example.features["character_pos_to_subword_pos"], dtype=np.int16)
        labels_mask = np.array(example.features["labels_mask"], dtype=np.bool_)
        labels = np.array(example.features["labels"], dtype=np.int16)
        spans = np.array(example.features["spans"], dtype=np.int16)
        return (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans)

    def collate_fn(self, batch):
        return collate_train_dataset(batch, pad_token_id=self.pad_token_id)


class CustomTestDataset(Dataset):
    def __init__(self, pad_token_id, examples, hyps_refs=None):
        self.examples = examples
        self.hyps_refs = hyps_refs
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        input_ids = np.array(example.features["input_ids"])
        input_mask = np.array(example.features["input_mask"])
        segment_ids = np.array(example.features["segment_ids"])
        input_ids_for_subwords = np.array(example.features["input_ids_for_subwords"])
        input_mask_for_subwords = np.array(example.features["input_mask_for_subwords"])
        segment_ids_for_subwords = np.array(example.features["segment_ids_for_subwords"])
        character_pos_to_subword_pos = np.array(example.features["character_pos_to_subword_pos"], dtype=np.int64)
        fragment_indices = np.array(example.features["fragment_indices"], dtype=np.int16)
        return (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            fragment_indices)

    def collate_fn(self, batch):
        return collate_test_dataset(batch, pad_token_id=self.pad_token_id)


def collate_test_dataset(batch, pad_token_id):
    max_length = max(len(sample[0]) for sample in batch)
    max_length_for_subwords = max(len(sample[3]) for sample in batch)
    max_length_for_fragment_indices = max(1, max(len(sample[7]) for sample in batch))

    input_ids_padded = []
    input_mask_padded = []
    segment_ids_padded = []
    input_ids_for_subwords_padded = []
    input_mask_for_subwords_padded = []
    segment_ids_for_subwords_padded = []
    character_pos_to_subword_pos_padded = []
    fragment_indices_padded = []

    for (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            fragment_indices,
    ) in batch:
        input_ids_padded.append(pad_or_copy(input_ids, max_length, pad_token_id))
        input_mask_padded.append(pad_or_copy(input_mask, max_length, 0))
        segment_ids_padded.append(pad_or_copy(segment_ids, max_length, 0))
        character_pos_to_subword_pos_padded.append(pad_or_copy(character_pos_to_subword_pos, max_length, 0))

        input_ids_for_subwords_padded.append(pad_or_copy(input_ids_for_subwords, max_length_for_subwords, pad_token_id))
        input_mask_for_subwords_padded.append(pad_or_copy(input_mask_for_subwords, max_length_for_subwords, 0))
        segment_ids_for_subwords_padded.append(pad_or_copy(segment_ids_for_subwords, max_length_for_subwords, 0))

        fragment_indices_padded.append(pad_or_copy_fragment_indices(fragment_indices, max_length_for_fragment_indices))

    return (
        torch.LongTensor(np.array(input_ids_padded)),
        torch.LongTensor(np.array(input_mask_padded)),
        torch.LongTensor(np.array(segment_ids_padded)),
        torch.LongTensor(np.array(input_ids_for_subwords_padded)),
        torch.LongTensor(np.array(input_mask_for_subwords_padded)),
        torch.LongTensor(np.array(segment_ids_for_subwords_padded)),
        torch.LongTensor(np.array(character_pos_to_subword_pos_padded)),
        torch.LongTensor(np.array(fragment_indices_padded)),
    )


def collate_train_dataset(batch, pad_token_id):
    max_length = max(len(sample[0]) for sample in batch)
    max_length_for_subwords = max(len(sample[3]) for sample in batch)
    max_length_for_spans = max(1, max(len(sample[8]) for sample in batch))

    input_ids_padded = []
    input_mask_padded = []
    segment_ids_padded = []
    input_ids_for_subwords_padded = []
    input_mask_for_subwords_padded = []
    segment_ids_for_subwords_padded = []
    character_pos_to_subword_pos_padded = []
    labels_mask_padded = []
    labels_padded = []
    spans_padded = []
    for (
            input_ids,
            input_mask,
            segment_ids,
            input_ids_for_subwords,
            input_mask_for_subwords,
            segment_ids_for_subwords,
            character_pos_to_subword_pos,
            labels_mask,
            labels,
            spans
    ) in batch:
        input_ids_padded.append(pad_or_copy(input_ids, max_length, pad_token_id))
        input_mask_padded.append(pad_or_copy(input_mask, max_length, 0))
        segment_ids_padded.append(pad_or_copy(segment_ids, max_length, 0))
        character_pos_to_subword_pos_padded.append(pad_or_copy(character_pos_to_subword_pos, max_length, 0))
        labels_mask_padded.append(pad_or_copy(labels_mask, max_length, 0))
        labels_padded.append(pad_or_copy(labels, max_length, 0))

        input_ids_for_subwords_padded.append(pad_or_copy(input_ids_for_subwords, max_length_for_subwords, pad_token_id))
        input_mask_for_subwords_padded.append(pad_or_copy(input_mask_for_subwords, max_length_for_subwords, 0))
        segment_ids_for_subwords_padded.append(pad_or_copy(segment_ids_for_subwords, max_length_for_subwords, 0))

        spans_padded.append(pad_or_copy_spans(spans, max_length_for_spans))

    return (
        torch.LongTensor(np.array(input_ids_padded)),
        torch.LongTensor(np.array(input_mask_padded)),
        torch.LongTensor(np.array(segment_ids_padded)),
        torch.LongTensor(np.array(input_ids_for_subwords_padded)),
        torch.LongTensor(np.array(input_mask_for_subwords_padded)),
        torch.LongTensor(np.array(segment_ids_for_subwords_padded)),
        torch.LongTensor(np.array(character_pos_to_subword_pos_padded)),
        torch.LongTensor(np.array(labels_mask_padded)),
        torch.LongTensor(np.array(labels_padded)),
        torch.LongTensor(np.array(spans_padded)),
    )


def pad_or_copy(data, target_length, pad_token):
    if len(data) < target_length:
        pad_length = target_length - len(data)
        return np.pad(data, pad_width=[0, pad_length], constant_values=pad_token)
    else:
        return data


def pad_or_copy_fragment_indices(fragment_indices, target_length):
    if len(fragment_indices) < target_length:
        padded_indices = np.zeros((target_length, 3), dtype=int)
        padded_indices[:, 1] = 1
        if len(fragment_indices) > 0:
            padded_indices[:fragment_indices.shape[0], :fragment_indices.shape[1]] = fragment_indices
        return padded_indices
    else:
        return fragment_indices


def pad_or_copy_spans(spans, target_length):
    if len(spans) < target_length:
        padded_indices = np.ones((target_length, 2), dtype=int) * -1
        if len(spans) > 0:
            padded_indices[:spans.shape[0], :spans.shape[1]] = spans
        return padded_indices
    else:
        return spans
