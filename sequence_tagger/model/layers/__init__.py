import tensorflow as tf


class Linear:

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

        with tf.variable_scope('Linear_Variables', reuse=tf.AUTO_REUSE):
            self.weights = tf.get_variable(
                'weights',
                dtype=tf.float32,
                shape=[2 * self.kwargs.get('hidden_size'), self.kwargs.get('tag_nums')]
            )
            self.bias = tf.get_variable(
                'bias',
                dtype=tf.float32,
                shape=[self.kwargs.get('tag_nums')],
                initializer=tf.zeros_initializer()
            )

    def __call__(self, inputs):
        with tf.name_scope('Linear'):
            sp = tf.shape(inputs)
            inputs = tf.reshape(inputs, [-1, 2 * self.kwargs.get('hidden_size')])
            logits = tf.matmul(inputs, self.weights) + self.bias
            logits = tf.reshape(logits, [-1, sp[1], self.kwargs.get('tag_nums')])

            return logits