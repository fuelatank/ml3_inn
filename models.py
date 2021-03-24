#TabNine::sem
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from env import Observation

def rnn(layer):
    def f(i, e, c, r, sizes):
        c = layers.Embedding(106, 4)(c)
        c = layers.Flatten()(c)
        e = layers.Embedding(106, 6)(e)
        e = layers.Flatten()(e)
        e = layers.Dense(64, activation='relu')(e)
        x = layers.Concatenate()([i, e, c, r])
        x = layers.Reshape((1, -1))(x)
        print(x.shape)
        for s in sizes:
            x = layer(s, stateful=True, return_sequences=True)(x)
            print(x.shape)
        x = layers.Flatten()(x)
        return x
    return f

def lstm(i, e, c, r, sizes):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    e = layers.Embedding(106, 6)(e)
    e = layers.Flatten()(e)
    e = layers.Dense(64, activation='relu')(e)
    x = layers.Concatenate()([i, e, c, r])
    x = layers.Reshape((1, -1))(x)
    print(x.shape)
    for s in sizes:
        x = layers.LSTM(s, stateful=True, return_sequences=True)(x)
        print(x.shape)
    x = layers.Flatten()(x)
    return x

def gru(i, e, c, r, sizes):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    e = layers.Embedding(106, 6)(e)
    e = layers.Flatten()(e)
    e = layers.Dense(64, activation='relu')(e)
    x = layers.Concatenate()([i, e, c, r])
    x = layers.Reshape((1, -1))(x)
    print(x.shape)
    for s in sizes:
        x = layers.GRU(s, stateful=True, return_sequences=True)(x)
        print(x.shape)
    x = layers.Flatten()(x)
    return x

def simpleModel(x):
    x = layers.Dense(128, activation='tanh')(x)
    x = layers.Dense(128, activation='tanh')(x)
    o = layers.Dense(120)(x)
    return o

def chooseOneCard(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(106)(x)
    return o

def chooseOneColor(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(11)(x) # opponent's board
    return o

def chooseYn(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(2)(x)
    return o

def chooseAnyCard(x, e, c):
    e = layers.Embedding(106, 4)(e)
    e = layers.Flatten()(e)
    x = layers.Concatenate()([x, e, c])
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(106)(x)
    return o

def chooseAnyColor(x, e, c):
    e = layers.Embedding(106, 4)(e)
    e = layers.Flatten()(e)
    x = layers.Concatenate()([x, e, c])
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(5)(x) # can't pass
    return o

def chooseAge(x):
    x = layers.Dense(32, activation='tanh')(x)
    x = layers.Dense(32, activation='tanh')(x)
    o = layers.Dense(10)(x)
    return o

def reveal(x):
    o = layers.Dense(1)(x)
    return o

rnns = {'lstm': lstm, 'gru': gru}

def buildModel(isize, esize, rnnSizes, rnn='lstm'):
    input1 = keras.Input(batch_shape=(1, isize))
    input2 = keras.Input(batch_shape=(1, esize))
    executingInput = keras.Input(batch_shape=(1, 1))
    chosenInput = keras.Input(batch_shape=(1, 105))
    revealInput = keras.Input(batch_shape=(1, 105))
    rnnfn = rnns[rnn] if rnn in rnns else None
    feature = rnnfn(input1, input2, executingInput, revealInput, rnnSizes)
    mainOutput = simpleModel(feature)
    oneCardOutput = chooseOneCard(feature, executingInput)
    oneColorOutput = chooseOneColor(feature, executingInput)
    anyCardOutput = chooseAnyCard(feature, executingInput, chosenInput)
    anyColorOutput = chooseAnyColor(feature, executingInput, chosenInput)
    ynOutput = chooseYn(feature, executingInput)
    ageOutput = chooseAge(feature)
    revealOutput = reveal(feature)
    outputs = [mainOutput, oneCardOutput, oneColorOutput, anyCardOutput, anyColorOutput, ynOutput, ageOutput, revealOutput]
    indexes = {}
    lastIndex = 0
    for o, k in zip(outputs, ['main', 'oc', 'ot', 'ac', 'at', 'yn', 'age', 'r']):
        indexes[k] = lastIndex
        lastIndex += o.shape[1]
    allOutput = layers.Concatenate()([mainOutput, oneCardOutput, oneColorOutput, anyCardOutput, anyColorOutput, ynOutput, ageOutput, revealOutput])

    rnnModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=feature)
    rnnLayers = rnnModel.layers[-1-len(rnnSizes):-1]
    assert len(rnnLayers) == len(rnnSizes)
    rnnModel.summary()

    model = keras.Model(inputs=[input1, input2, executingInput, chosenInput, revealInput],
            outputs=allOutput)
    model.summary()
    return model, rnnLayers, indexes

def modelTFFunction(model):
    @tf.function
    def pred(data, denseValids):
        #print(model.name)
        r = model(data)[0]
        return tf.exp(r) * denseValids
    return pred

def modelTFFunctionAction(model):
    @tf.function
    def pred(data, act):
        return model(data)[0][act]
    return pred

class QModel:
    def __init__(self, isize, esize, rnnSizes, lr, rnn='lstm', gamma=0.99):
        self.model, self.rnnLayers, self.indexes = buildModel(isize, esize, rnnSizes, rnn=rnn)
        self.func = modelTFFunction(self.model)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.target = Target(buildModel(isize, esize, rnnSizes, rnn=rnn))
        self.updateTarget()

    def fit(self, data, target=None, loops=1, double=True):
        if not target:
            target = self.target
        for _ in range(loops):
            #print('loop', loop)
            for i, episode in enumerate(data):
                print('episode', i+1, 'length', len(episode))
                t0 = time.time()
                ys = []
                xs = []
                self.predict(episode[0][0])
                target.predict(episode[0][0])
                for obs, act, reward, nextObs, n_done in episode:
                    if n_done:
                        nqs = target.predict(nextObs)
                        if double:
                            cqs = self.predict(nextObs)
                            bestAct = tf.argmax(cqs)
                            q_next = nqs[bestAct]
                        else:
                            q_next = tf.reduce_max(nqs)
                        ys.append(self.gamma * q_next)
                    else:
                        ys.append(reward)
                    xs.append((obs, act))
                self.reset_states()
                target.reset_states()
                print('predict:', time.time()-t0)
                t0 = time.time()
                with tf.GradientTape() as tape:
                    y_preds = []
                    for obs, act in xs:
                        model = self.model
                        y_pred = model(obs.data)[0][act]
                        y_preds.append(y_pred)
                    loss = tf.reduce_mean((tf.stack(ys) - tf.stack(y_preds)) ** 2)
                grads = tape.gradient(loss, self.model.trainable_weights)
                self.optimizer.apply_gradients(
                    (grad, var) 
                    for (grad, var) in zip(grads, self.model.trainable_variables) 
                    if grad is not None)
                self.reset_states()
                print('train:', time.time()-t0)

    def step(self, obs):
        model = self.model
        r = model(obs.data)[0]
        return self.argmaxWithValidFilter(r, obs.valids)

    def predict_slow(self, obs):
        model = self.model
        r = model(obs.data)[0]
        r = self.validFilter(r, obs.valids)
        return r

    def predict(self, obses):
        dense = np.zeros((len(obses),) + self.model.output_shape[1:])
        for i, obs in enumerate(obses):
            index = self.indexes[obs.type]
            dense[i, [index + v for v in obs.valids]] = 1
        r = self.func(obs.data, tf.constant(dense, dtype=tf.float32))
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
        #self.modelsDict = dict_
        #self.rnnLayers = rnnLayers

    def set(self, model):
        self.model.set_weights(model.get_weights())
    
    def reset_states(self):
        for layer in self.rnnLayers:
            layer.reset_states()

    def predict(self, obs):
        model = self.modelsDict[obs.type]
        r = model(obs.data)[0]
        r = self.validFilter(r, obs.valids)
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