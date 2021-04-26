import numpy as np
import random

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
    
    def noise_copy(self, scale=0.001, num_layers=1):
        noise_layers = random.sample(self.weights, k=num_layers)
        new_weights = [w + np.random.normal(0, scale, w.shape) \
            if w in noise_layers else w.copy() \
            for w in self.weights]
        return NPWeights(self.names, new_weights)
    
    def random_copy(self, scale=0.001):
        return NPWeights(self.names, [np.random.normal(0, scale, w.shape) for w in self.weights])