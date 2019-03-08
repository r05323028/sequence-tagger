import numpy as np


def dump_sentence(sent):
    char, word, labels = [], [], []

    for w, l in sent:
        word.append(w)
        labels.append(l)
        char.append(list(w))

    return char, word, labels

def stack_features(batch, prediction=False):
    char, word, labels = [], [], []

    for row in batch:
        char.append(row[0])
        word.append(row[1])
        if not prediction:
            labels.append(row[2])

    if not prediction:
        return char, word, labels

    return char, word

def get_max_sentence_length(sequences):
    lengths = [len(seq) for seq in sequences]

    return max(lengths)

def get_max_word_length(sequences):
    lengths = [[len(word) for word in seq] for seq in sequences]
    lengths = [max(l) if l else 0 for l in lengths]

    return max(lengths)

def pad_sentence_sequences(sequences, max_len, value):
    return [
        seq + [value] * (max_len - len(seq))
        for seq in sequences
    ]

def pad_word_sequences(sequences, max_word_len, max_sent_len, value):
    return [
        [word + [value] * (max_word_len - len(word)) for word in sent] + [[value] * max_word_len] * (max_sent_len - len(sent))  for sent in sequences
    ]

def get_sentence_sequence_lengths(sequences):
    return [len(seq) for seq in sequences]

def get_word_sequence_lengths(sequences):
    return [[len(word) for word in seq] for seq in sequences]

def pad_word_sequence_lengths(sequences, max_sent_len):
    return [
        np.pad(seq, pad_width=(0, max_sent_len - len(seq)), mode='constant', constant_values=0)
        for seq in sequences]