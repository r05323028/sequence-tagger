import random
import fastText
import numpy as np
import tensorflow as tf
from sequence_tagger.utils import (
    dump_sentence, 
    stack_features, 
    get_max_word_length, 
    get_max_sentence_length, 
    pad_word_sequences,
    pad_sentence_sequences, 
    get_sentence_sequence_lengths,
    get_word_sequence_lengths,
    pad_word_sequence_lengths)
from sequence_tagger.utils.embedding import Embedding

class BatchGenerator:
    
    def __init__(self, dataset, **kwargs):
        self._dataset = dataset
        self._batch_size = kwargs.get('batch_size', 32)
        self._batch_nums = int(len(self._dataset) / self._batch_size) + 1

    def __call__(self):
        random.shuffle(self._dataset)
        for batch in self._make_batch():
            yield batch

    @property
    def batch_nums(self):
        return self._batch_nums

    def _make_batch(self):
        start, end = 0, self._batch_size

        for _ in range(self._batch_nums):
            temp_rows = self._dataset[start:end]
            temp_rows = [
                dump_sentence(row) for row in temp_rows]

            yield temp_rows
            start, end = end, end + self._batch_size

class BatchFormatter:

    def __init__(self, *args, **kwargs):
        self._pad = '<pad>'
        self._char_embedding = Embedding(kwargs.get('char_embedding_path'))
        self._word_embedding = Embedding(kwargs.get('word_embedding_path'))

    def __call__(self, batch, prediction=False):
        if prediction:
            char, word = stack_features(batch, prediction=True)
        
        else:
            char, word, labels = stack_features(batch)

        # get max length
        max_char_len = get_max_word_length(char)
        max_word_len = max_labels_len = get_max_sentence_length(word)

        # get sequence length for dynamic rnn
        char_seq_lengths = get_word_sequence_lengths(char)
        word_seq_lengths = get_sentence_sequence_lengths(word)
        char_seq_lengths = pad_word_sequence_lengths(char_seq_lengths, max_word_len)

        # pad sequences
        char_pad = pad_word_sequences(char, max_char_len, max_word_len, self._pad)
        word_pad = pad_sentence_sequences(word, max_word_len, self._pad)
        
        char_encoded = self._char_encode(char_pad, self._char_embedding)
        word_encoded = self._word_encode(word_pad, self._word_embedding)

        if not prediction:
            labels_pad = pad_sentence_sequences(labels, max_labels_len, 0)

            return (
                np.array(char_encoded), np.array(word_encoded), np.array(char_seq_lengths), np.array(word_seq_lengths), np.array(labels_pad))
        else:
            return (
                np.array(char_encoded), np.array(word_encoded), np.array(char_seq_lengths), np.array(word_seq_lengths))

    def _char_encode(self, padded_sequences, embedding):
        return [[[embedding[c] for c in word] for word in seq] for seq in padded_sequences]

    def _word_encode(self, padded_sequences, embedding):
        return [[embedding[s] for s in seq] for seq in padded_sequences]




        