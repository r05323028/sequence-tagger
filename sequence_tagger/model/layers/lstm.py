import tensorflow as tf


class Char_BiLSTM:

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

        with tf.variable_scope('Char_LSTM_Variables', reuse=tf.AUTO_REUSE):
            self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.kwargs.get('hidden_size'))
            self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.kwargs.get('hidden_size'))

    def __call__(self, char_sequence, char_sequence_lengths):
        with tf.name_scope('Char_LSTM'):
            sp = tf.shape(char_sequence)
            char_sequence_lengths = tf.reshape(char_sequence_lengths, shape=[-1])
            char_sequence = tf.reshape(
                char_sequence, [sp[0] * sp[1], sp[2], self.kwargs.get('embedding_size')])
            _, ((_, fw_output), (_, bw_output)) = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell, 
                self.bw_cell, 
                char_sequence, 
                sequence_length=char_sequence_lengths, 
                dtype=tf.float32, 
                scope='char_bilstm')
            output = tf.concat([fw_output, bw_output], axis=-1)
            output = tf.reshape(output, [sp[0], sp[1], 2 * self.kwargs.get('hidden_size')])

            return output

class Word_BiLSTM:

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

        with tf.variable_scope('Word_LSTM_Variables', reuse=tf.AUTO_REUSE):
            self.fw_cell = tf.nn.rnn_cell.LSTMCell(self.kwargs.get('hidden_size'))
            self.bw_cell = tf.nn.rnn_cell.LSTMCell(self.kwargs.get('hidden_size'))

    def __call__(self, word_sequence, word_sequence_lengths):
        with tf.name_scope('Word_LSTM'):
            (fw_output, bw_output), _ = tf.nn.bidirectional_dynamic_rnn(
                self.fw_cell,
                self.bw_cell,
                word_sequence,
                dtype=tf.float32,
                scope='word_bilstm'
            )
            output = tf.concat([fw_output, bw_output], axis=-1)

            return output
