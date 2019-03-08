import os
import abc
import numpy as np
import tf_metrics
import tensorflow as tf
from functools import wraps
from sequence_tagger.model.layers import Linear
from sequence_tagger.model.layers.lstm import Char_BiLSTM, Word_BiLSTM
from sequence_tagger.model.layers.crf import CRF


class BaseModel(metaclass=abc.ABCMeta):

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs
        self.sess, self.saver = None, None
        self.train_step, self.test_step = 0, 0
        self.model_path = os.path.join(
            self.kwargs.get('model_dir'), self.kwargs.get('model_name'))

    @abc.abstractmethod
    def _build_model(self):
        return NotImplemented

    @abc.abstractmethod
    def train_on_batch(self):
        return NotImplemented

    @abc.abstractmethod
    def predict_on_batch(self):
        return NotImplemented

    def save_model(self):
        self.saver.save(
            self.sess, self.model_path)

    def load_model(self):
        model = self.saver.restore(
            self.sess, self.kwargs.get('model_dir'))

        return model

class NER(BaseModel):

    def __init__(self, *args, **kwargs):
        super(NER, self).__init__(*args, **kwargs)

        self._init_layers()
        self._build_model()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.train_log_path = os.path.join(
            self.kwargs.get('log_dir'), 'train')
        self.test_log_path = os.path.join(self.kwargs.get('log_dir'), 'test')
        
        self.merged = tf.summary.merge_all()
        self.train_log_writer = tf.summary.FileWriter(self.train_log_path, self.sess.graph)
        self.test_log_writer = tf.summary.FileWriter(self.test_log_path, self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())

    def _build_placeholders(self):
        self.char_sequence = tf.placeholder(
            tf.float32, shape=[None, None, None, self.kwargs.get('embedding_size')])
        self.word_sequence = tf.placeholder(
            tf.float32, shape=[None, None, self.kwargs.get('embedding_size')])
        self.char_sequence_length = tf.placeholder(
            tf.int32, shape=[None, None])
        self.word_sequence_length = tf.placeholder(
            tf.int32, shape=[None])
        self.labels = tf.placeholder(
            tf.int32, shape=[None, None])

    def _init_layers(self):
        self.char_bilstm = Char_BiLSTM(*self.args, **self.kwargs)
        self.word_bilstm = Word_BiLSTM(*self.args, **self.kwargs)
        self.linear = Linear(*self.args, **self.kwargs)
        self.crf = CRF(*self.args, **self.kwargs)

    def _build_model(self):
        with tf.name_scope('input_layer'):
            self._build_placeholders()

        with tf.name_scope('hidden_layers'):
            char_representation = self.char_bilstm(
                self.char_sequence, self.char_sequence_length)
            concat_representation = tf.concat(
                [self.word_sequence, char_representation], axis=-1)
            word_representation = self.word_bilstm(
                concat_representation, self.word_sequence_length)

        with tf.name_scope('output_layer'):
            self.logits = self.linear(word_representation)
            loglikelihood, self.trans_params = self.crf(
                self.logits, self.labels, self.word_sequence_length)
            self.loss = tf.reduce_mean(-loglikelihood)

        with tf.name_scope('prediction'):
            self.preds = self.crf.decode(
                self.logits, self.trans_params, self.word_sequence_length)

        with tf.name_scope('metrics'):
            p, self.p_op = tf_metrics.precision(
                self.labels, self.preds, 2, pos_indices=[1])
            r, self.r_op = tf_metrics.recall(
                self.labels, self.preds, 2, pos_indices=[1])
            f1, self.f1_op = tf_metrics.f1(
                self.labels, self.preds, 2, pos_indices=[1])

            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('precision', p)
            tf.summary.scalar('recall', r)
            tf.summary.scalar('f1', f1)

        with tf.name_scope('optimizer'):
            opt = tf.train.AdamOptimizer(learning_rate=self.kwargs.get('learning_rate'))
            self.train_op = opt.minimize(self.loss)

    def train_on_batch(self, char_sequence, word_sequence, char_sequence_length, word_sequence_length, labels):
        self.train_step += 1

        self.sess.run(
            [self.train_op],
            feed_dict={
                self.char_sequence: char_sequence,
                self.word_sequence: word_sequence,
                self.char_sequence_length: char_sequence_length,
                self.word_sequence_length: word_sequence_length,
                self.labels: labels
            }
        )

        train_summary, _, _, _ = self.sess.run(
            [self.merged, self.p_op, self.r_op, self.f1_op],
            feed_dict={
                self.char_sequence: char_sequence,
                self.word_sequence: word_sequence,
                self.char_sequence_length: char_sequence_length,
                self.word_sequence_length: word_sequence_length,
                self.labels: labels
            }
        )

        self.train_log_writer.add_summary(train_summary, global_step=self.train_step)

    def predict_on_batch(self, char_sequence, word_sequence, char_sequence_length, word_sequence_length):
        return self.sess.run(
            self.preds,
            feed_dict={
                self.char_sequence: char_sequence,
                self.word_sequence: word_sequence,
                self.char_sequence_length: char_sequence_length,
                self.word_sequence_length: word_sequence_length,
            }
        )

    def test_on_batch(self, char_sequence, word_sequence, char_sequence_length, word_sequence_length, labels):
        self.test_step += 1

        test_summary, _, _, _ = self.sess.run(
            [self.merged, self.p_op, self.r_op, self.f1_op],
            feed_dict={
                self.char_sequence: char_sequence,
                self.word_sequence: word_sequence,
                self.char_sequence_length: char_sequence_length,
                self.word_sequence_length: word_sequence_length,
                self.labels: labels
            }
        )

        self.test_log_writer.add_summary(test_summary, global_step=self.test_step)