import fastText
import numpy as np


class Embedding:

    def __init__(self, fname, **kwargs):
        self.model = fastText.load_model(fname)
        self.dim = self.model.get_dimension()
        self.oov = {}

    def __getitem__(self, key):
        try:
            if key == '<pad>':
                return np.zeros(shape=self.dim)
                
            return self.model.get_word_vector(key)

        except KeyError:
            self.oov[key] = np.random.uniform(-0.25, 0.25, size=self.dim)
            return self.oov[key]