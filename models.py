#TabNine::sem
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseObs import Observation
from obs import stackObs
from weights import NPWeights
import trainconfig as tc

rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}
activations = {'relu': F.relu, 'tanh': F.tanh}

class ChooseModel(nn.Module):
    def __init__(self, inputSize, outputSize, executing=False, chosen=False, hidden=32, activation=tc.ACTIVATION):
        super(ChooseModel, self).__init__()
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.executing = executing
        self.chosen = chosen
        self.a = activations[activation]
        extraSize = 0
        if self.executing:
            self.embed = nn.Embedding(106, 4)
            extraSize += 4
        if self.chosen:
            self.chosenLinear = nn.Linear(105, 16)
            extraSize += 16
        self.linear1 = nn.Linear(self.inputSize+extraSize, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, self.outputSize)
    
    def forward(self, x):
        if self.chosen:
            mainInput, eInput, cInput = x
            e = self.embed(eInput)#F.relu/tanh!
            c = self.a(self.chosenLinear(cInput))
            x = torch.cat([mainInput, e, c], dim=1)
        elif self.executing:
            mainInput, eInput = x
            e = self.embed(eInput)
            x = torch.cat([mainInput, e], dim=1)
        x = self.a(self.linear1(x))
        x = self.a(self.linear2(x))
        x = self.linear3(x)
        return x

class BasicModel(nn.Module):
    def __init__(self, isize, esize, rnnSizes, rnn='lstm', activation=tc.ACTIVATION):
        super(BasicModel, self).__init__()
        self.isize = isize
        self.esize = esize
        self.rnnSizes = rnnSizes
        self.rnnFunc = rnns[rnn]
        self.a = activations[activation]
        self.buildModel()
    
    def buildModel(self):
        self.xEmbed = nn.Embedding(106, 4)
        self.eEmbed = nn.Embedding(106, 6)
        self.eLinear = nn.Linear(self.esize*6, 48)
        self.rLinear = nn.Linear(105, 16)
        self.rnnLayers = []
        rnnInputLength = 48+16+self.isize+4
        for size in self.rnnSizes:
            l = nn.LSTM(rnnInputLength, size)
            self.rnnLayers.append(l)
            rnnInputLength = size
        self.main = ChooseModel(rnnInputLength, 120, hidden=128, activation=self.a)
        self.oc = ChooseModel(rnnInputLength, 106, executing=True, activation=self.a)
        self.ot = ChooseModel(rnnInputLength, 11, executing=True, activation=self.a)
        self.ac = ChooseModel(rnnInputLength, 106, executing=True, chosen=True, activation=self.a)
        self.at = ChooseModel(rnnInputLength, 5, executing=True, chosen=True, activation=self.a)
        self.yn = ChooseModel(rnnInputLength, 2, executing=True, activation=self.a)
        self.age = ChooseModel(rnnInputLength, 10, activation=self.a)
        self.r = ChooseModel(rnnInputLength, 1, hidden=1, activation=self.a)
    
    def forward(self, x, hs=None):
        normal, embedRequired, executing, chosen, reveal = x
        x = self.xEmbed(executing)
        x = x.view(x.size()[0], -1)
        e = self.eEmbed(embedRequired)
        e = e.view(e.size()[0], -1)
        e = self.a(self.eLinear(e))
        r = self.a(self.rLinear(reveal))
        x = torch.cat([normal, e, x, r])
        x = torch.unsqueeze(x, 1)
        nhs = []
        if torch.is_grad_enabled():
            for l in self.rnnLayers:
                x, nh = l(x)
                nhs.append(nh)
        else:
            for h, l in zip(hs, self.rnnLayers):
                x, nh = l(x, h)
                nhs.append(nh)
        x = x.view(x.size()[0], -1)
        main = self.main(x)
        oc = self.oc(x, executing)
        ot = self.ot(x, executing)
        ac = self.ac(x, executing, chosen)
        at = self.at(x, executing, chosen)
        yn = self.yn(x, executing)
        age = self.age(x)
        r = self.r(x)
        x = torch.cat([main, oc, ot, ac, at, yn, age, r], 1)
        return x, nhs

def rnn(layer):
    def f(i, e, c, r, sizes, training=False):
        c = layers.Embedding(106, 4)(c)
        c = layers.Flatten()(c)
        e = layers.Embedding(106, 6)(e)
        e = layers.Flatten()(e)
        e = layers.Dense(48, activation=tc.ACTIVATION)(e)
        r = layers.Dense(16, activation=tc.ACTIVATION)(r)
        x = layers.Concatenate()([i, e, c, r])
        x = layers.Reshape((1, -1))(x)
        print(x.shape)
        for s in sizes:
            x = layer(s, stateful=(not training), return_sequences=True)(x)
            print(x.shape)
        x = layers.Flatten()(x)
        return x
    return f

rnns = {'lstm': lstm, 'gru': gru}

def buildModel(isize, esize, rnnSizes, rnn='lstm'):
    model = BasicModel(isize, esize,  rnnSizes, rnn=rnn)
    return model

def validFilter(logits, denseValids):
    return torch.exp(logits) * denseValids

def modelTFFunctionAction(model):
    @tf.function
    def pred(data, act):
        return model(data)[0][act]
    return pred

class PolicyModel:
    def __init__(self, isize, esize, rnnSizes, lr, rnn='lstm'):
        self.model = buildModel(isize, esize, rnnSizes, rnn=rnn)
    
    def step(self, obs):
        probs = self._step(obs.data, obs.valids).numpy()
        action = np.random.choice(361, p=probs)
        return action
    
    def _step(self, data, valids):
        with torch.no_grad():
            logits, self.states = self.model({'x': data, 'hs': self.states})
            r = validFilter(logits, torch.unsqueeze(valids, 0))
            probs = r / torch.sum(r)
        return probs
    
    def get_weights(self):
        return NPWeights.from_keras_model(self.model)
    
    def set_weights(self, w):
        self.model.set_weights(w.weights)
    
    def reset_states(self):
        self.states = None

class QModel:
    def __init__(self, isize, esize, rnnSizes, lr, rnn='lstm', gamma=0.99):
        self.model, self.rnnLayers, self.indexes = buildModel(isize, esize, rnnSizes, rnn=rnn)
        self.fitmodel, _, _ = buildModel(isize, esize, rnnSizes, rnn=rnn, training=True)
        self.fitmodel.set_weights(self.model.get_weights())
        self.func = modelTFFunction(self.fitmodel)
        self.stepfunc = modelTFFunction(self.model)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.target = Target(buildModel(isize, esize, rnnSizes, rnn=rnn, training=True))
        self.updateTarget()
        self._fit = self.fit_maker()

    def fit(self, data, target=None, loops=1, double=True):
        if not target:
            target = self.target
        for _ in range(loops):
            #print('loop', loop)
            for _, episode in enumerate(data):
                #print(len(episode), end=' ')
                #t0 = time.time()
                acts = tf.constant([s[1] for s in episode], dtype=tf.int32)
                rew = tf.constant([episode[-1][2]], dtype=tf.float32)
                allObs, allValids = stackObs(episode)
                self._fit(allObs, allValids, acts, rew, double=tf.constant(double, dtype=tf.bool))
                self.fitmodel.set_weights(self.model.get_weights())
    
    def fit_maker(self):
        @tf.function(input_signature=(
            [tf.TensorSpec(shape=s, dtype=tf.float32) for s in self.fitmodel.input_shape],
            tf.TensorSpec(shape=(None, 361), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.int32),
            tf.TensorSpec(shape=(1), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.bool)))
        def f(allObs, allValids, acts, rew, double):
            target = self.target
            nqs = target.predict(allObs, allValids)[1:]
            if double:
                cqs = self.predict(allObs, allValids)[1:]
                bestAct = tf.argmax(cqs, axis=-1)
                rg = tf.range(len(bestAct), dtype=tf.int64)
                indices = tf.stack([rg, bestAct], axis=-1)
                q_next = tf.gather_nd(nqs, indices)
            else:
                q_next = tf.reduce_max(nqs, axis=-1)
            ys = self.gamma * tf.math.log(q_next)
            ys = tf.concat([ys, rew], axis=0)
            with tf.GradientTape() as tape:
                qs = self.fitmodel(allObs)
                rg = tf.range(tf.shape(acts)[0])#, dtype=tf.int64)
                indices = tf.stack([rg, acts], axis=-1)
                y_preds = tf.gather_nd(qs, indices)
                loss = tf.reduce_mean((tf.stack(ys) - tf.stack(y_preds)) ** 2)
            grads = tape.gradient(loss, self.fitmodel.trainable_weights)
            self.optimizer.apply_gradients(
                (grad, var) 
                for (grad, var) in zip(grads, self.model.trainable_variables) 
                if grad is not None)
        return f

    def step_slow(self, obs):
        r = self.stepfunc(obs.data, tf.expand_dims(obs.valids, axis=0))[0]
        return tf.argmax(r)
    
    def step(self, obs):
        return self._step(obs.data, obs.valids)
    
    @tf.function
    def _step(self, data, valids):
        r = self.stepfunc(data, tf.expand_dims(valids, axis=0))[0]
        return tf.argmax(r)

    def predict_slow(self, obs):
        model = self.model
        r = model(obs.data)[0]
        r = self.validFilter(r, obs.valids)
        return r

    @tf.function
    def predict(self, data, valids):
        r = self.func(data, valids)
        return r

    def updateTarget(self):
        self.target.set(self.model)

    def validFilter(self, output, valids, log=False):
        #output = output.numpy()
        #dense = np.zeros(output.shape)
        #dense[valids] = 1
        dense = valids
        if log:
            dense = np.log(dense)
            return output + dense
        else:
            return tf.exp(output) * dense

    def argmaxWithValidFilter(self, output, valids):
        return tf.argmax(self.validFilter(output, valids), axis=-1)

    def reset_states(self):
        for layer in self.rnnLayers:
            layer.reset_states()

class Target:
    def __init__(self, model):
        self.model, self.modelsDict, self.rnnLayers = model
        self.func = modelTFFunction(self.model)
        #self.modelsDict = dict_
        #self.rnnLayers = rnnLayers

    def set(self, model):
        self.model.set_weights(model.get_weights())
    
    def reset_states(self):
        for layer in self.rnnLayers:
            layer.reset_states()

    @tf.function
    def predict(self, data, valids):
        r = self.func(data, valids)
        return r

    def validFilter(self, output, valids, log=False):
        output = output.numpy()
        dense = np.zeros(output.shape)
        dense[valids] = 1
        if log:
            return output + valids[-1]
        else:
            return tf.exp(output) * dense

'''qmodel = QModel(300, 20, [128, 128], 1e-4, rnn='gru')
i = tf.random.normal((1, 300))
e = tf.random.categorical(tf.zeros((1, 106)), num_samples=20)
v = tf.cast(tf.random.categorical(tf.constant([[0., 0.]]), num_samples=120), tf.float32)
print(i.shape, e.shape, v)
obs = Observation([i, e, tf.constant([[45]])], v, 'main')
print(qmodel.predict(obs), qmodel.step(obs), qmodel.step(obs), qmodel.step(obs))'''