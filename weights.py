import numpy as np
import random

import trainconfig as tc

class NPWeights:
    def __init__(self, names, weights):
        self.names = names
        self.weights = weights
    
    @staticmethod
    def from_keras_model(model):
        names = [w.name for w in model.weights]
        weights = model.get_weights()
        return NPWeights(names, weights)
    
    def copy(self):
        return NPWeights(self.names, [w.copy() for w in self.weights])
    
    def noise_copy(self, scale=tc.WEIGHT_SCALE, num_layers=tc.NUM_MUTATE_LAYERS):
        noise_layers = random.sample(self.names, k=num_layers)
        new_weights = [w + np.random.normal(0, scale, w.shape) \
            if n in noise_layers else w.copy() \
            for n, w in zip(self.names, self.weights)]
        return NPWeights(self.names, new_weights)
    
    def random_copy(self, scale=tc.WEIGHT_SCALE_INIT):
        return NPWeights(self.names, [np.random.normal(0, scale, w.shape) for w in self.weights])