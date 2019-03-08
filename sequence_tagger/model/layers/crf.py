import tensorflow as tf


class CRF:

    def __init__(self, *args, **kwargs):
        self.args, self.kwargs = args, kwargs

    def __call__(self, logits, labels, sequence_length):
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(
            logits, labels, sequence_length
        )

        return log_likelihood, trans_params

    def decode(self, logits, trans_params, sequence_length):
        preds, _ = tf.contrib.crf.crf_decode(
            logits, trans_params, sequence_length)

        return preds
