import pickle
from tqdm import tqdm, trange
from argparse import ArgumentParser
from sequence_tagger.model import NER
from sequence_tagger.utils.batch import BatchFormatter, BatchGenerator
from sequence_tagger.utils.embedding import Embedding

def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        '--batch-size', type=int, dest='batch_size', default=32)
    arg_parser.add_argument(
        '--epoch-nums', type=int, dest='epoch_nums', default=10)
    arg_parser.add_argument(
        '--learning-rate', type=float, dest='learning_rate', default=0.001)
    arg_parser.add_argument(
        '--hidden-size', type=int, dest='hidden_size', default=128)
    arg_parser.add_argument(
        '--tag-nums', type=int, dest='tag_nums', default=2)
    arg_parser.add_argument(
        '--char-embedding-path', type=str, dest='char_embedding_path', default='datasets/char_embeddings.bin')
    arg_parser.add_argument(
        '--word-embedding-path', type=str, dest='word_embedding_path', default='datasets/word_embeddings.bin')
    arg_parser.add_argument(
        '--embedding-size', type=int, dest='embedding_size', default=100)
    arg_parser.add_argument(
        '--train-fp', type=str, dest='train_fp', default='datasets/train.pkl')
    arg_parser.add_argument(
        '--test-fp', type=str, dest='test_fp', default='datasets/test.pkl')
    arg_parser.add_argument(
        '--model-dir', type=str, dest='model_dir', default='outputs/')
    arg_parser.add_argument(
        '--model-name', type=str, dest='model_name', default='crf')
    arg_parser.add_argument(
        '--log-dir', type=str, dest='log_dir', default='logs')
    args = arg_parser.parse_args()

    return vars(args)

def load_dataset(fname):
    with open(fname, 'rb') as file:
        data = pickle.load(file)

        return data

def main(**args):
    batch_formatter = BatchFormatter(**args)
    train_data = load_dataset(args.get('train_fp'))
    train_batch_generator = BatchGenerator(train_data, **args)
    test_data = load_dataset(args.get('test_fp'))
    test_batch_generator = BatchGenerator(test_data, **args)

    model = NER(**args)

    for _ in trange(args.get('epoch_nums'), desc='epoch'):

        for batch in tqdm(train_batch_generator(), total=train_batch_generator.batch_nums, desc='training'):
            char_seq, word_seq, char_seq_len, word_seq_len, labels = batch_formatter(batch)
            model.train_on_batch(
                char_sequence=char_seq,
                word_sequence=word_seq,
                char_sequence_length=char_seq_len,
                word_sequence_length=word_seq_len,
                labels=labels)

        for batch in tqdm(test_batch_generator(), total=test_batch_generator.batch_nums, desc='testing'):
            char_seq, word_seq, char_seq_len, word_seq_len, labels = batch_formatter(batch)
            model.test_on_batch(
                char_sequence=char_seq,
                word_sequence=word_seq,
                char_sequence_length=char_seq_len,
                word_sequence_length=word_seq_len,
                labels=labels)

if __name__ == "__main__":
    args = get_args()
    main(**args)