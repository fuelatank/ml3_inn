#TabNine::sem
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from env import Observations

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

    rnnModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=feature)
    rnnLayers = rnnModel.layers[-1-len(rnnSizes):-1]
    assert len(rnnLayers) == len(rnnSizes)
    rnnModel.summary()

    mainModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=mainOutput)
    oneCardModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=oneCardOutput)
    oneColorModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=oneColorOutput)
    anyCardModel = keras.Model(inputs=[input1, input2, executingInput, revealInput, chosenInput], outputs=anyCardOutput)
    anyColorModel = keras.Model(inputs=[input1, input2, executingInput, revealInput, chosenInput], outputs=anyColorOutput)
    ynModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=ynOutput)
    ageModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=ageOutput)
    revealModel = keras.Model(inputs=[input1, input2, executingInput, revealInput], outputs=revealOutput)

    #models = [mainModel, ynModel, oneCardModel, oneColorModel, anyCardModel, anyColorModel]
    modelsDict = {'main': mainModel,'yn': ynModel,
        'oc': oneCardModel, 'ot': oneColorModel,
        'ac': anyCardModel, 'at': anyColorModel,
        'age': ageModel, 'r': revealModel}
    model = keras.Model(inputs=[input1, input2, executingInput, chosenInput, revealInput],
            outputs=[mainOutput, oneCardOutput, oneColorOutput, anyCardOutput, anyColorOutput, ynOutput, ageOutput, revealOutput])
    model.summary()
    return model, modelsDict, rnnLayers

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
        '''input1 = keras.Input(batch_shape=(1, isize))
        input2 = keras.Input(batch_shape=(1, esize))
        executingInput = keras.Input(batch_shape=(1, 1))
        chosenInput = keras.Input(batch_shape=(1, 105))

        self.rnn = rnn
        rnnfn = rnns[rnn] if rnn in rnns else None
        self.feature = rnnfn(input1, input2, executingInput, rnnSizes)
        mainOutput = simpleModel(self.feature)
        oneCardOutput = chooseOneCard(self.feature, executingInput)
        oneColorOutput = chooseOneColor(self.feature, executingInput)
        anyCardOutput = chooseAnyCard(self.feature, executingInput, chosenInput)
        anyColorOutput = chooseAnyColor(self.feature, executingInput, chosenInput)
        ynOutput = chooseYn(self.feature, executingInput)

        self.rnnModel = keras.Model(inputs=[input1, input2, executingInput], outputs=self.feature)
        self.rnnLayers = self.rnnModel.layers[9:-1]
        assert len(self.rnnLayers) == len(rnnSizes)
        self.rnnModel.summary()

        self.mainModel = keras.Model(inputs=[input1, input2, executingInput], outputs=mainOutput)
        self.oneCardModel = keras.Model(inputs=[input1, input2, executingInput], outputs=oneCardOutput)
        self.oneColorModel = keras.Model(inputs=[input1, input2, executingInput], outputs=oneColorOutput)
        self.anyCardModel = keras.Model(inputs=[input1, input2, executingInput, chosenInput], outputs=anyCardOutput)
        self.anyColorModel = keras.Model(inputs=[input1, input2, executingInput, chosenInput], outputs=anyColorOutput)
        self.ynModel = keras.Model(inputs=[input1, input2, executingInput], outputs=ynOutput)

        self.models = [self.mainModel, self.ynModel, self.oneCardModel, self.oneColorModel, self.anyCardModel, self.anyColorModel]
        self.modelsDict = {'main': self.mainModel,'yn': self.ynModel,
        'oc': self.oneCardModel, 'ot': self.oneColorModel,
        'ac': self.anyCardModel, 'at': self.anyColorModel}
        self.model = keras.Model(inputs=[input1, input2, executingInput, chosenInput],
            outputs=[mainOutput, oneCardOutput, oneColorOutput, anyCardOutput, anyColorOutput, ynOutput])
        self.model.summary()'''
        self.model, self.modelsDict, self.rnnLayers = buildModel(isize, esize, rnnSizes, rnn=rnn)
        self.functionDict = {}
        for k in self.modelsDict:
            model = self.modelsDict[k]
            @tf.function
            def pred(data, denseValids):
                r = model(data)[0]
                return tf.exp(r) * denseValids
            self.functionDict[k] = pred
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.gamma = gamma
        self.target = Target(buildModel(isize, esize, rnnSizes, rnn=rnn))
        self.updateTarget()

    def fit(self, data, target=None, loops=1, double=True):
        if not target:
            target = self.target
        for loop in range(loops):
            print('loop', loop)
            for i, episode in enumerate(data):
                print('episode', i+1)
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
                            q_next = tf.maximum(nqs)
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
                        model = self.modelsDict[obs.type]
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
        model = self.modelsDict[obs.type]
        r = model(obs.data)[0]
        return self.argmaxWithValidFilter(r, obs.valids)

    def predict_slow(self, obs):
        model = self.modelsDict[obs.type]
        r = model(obs.data)[0]
        r = self.validFilter(r, obs.valids)
        return r

    def predict(self, obs):
        dense = np.zeros(self.modelsDict[obs.type].output_shape[1:])
        dense[obs.valids] = 1
        return self.functionDict[obs.type](obs.data, tf.constant(dense, dtype=tf.float32))

    def updateTarget(self):
        self.target.set(self.model)

    def validFilter(self, output, valids, log=False):
        #output = output.numpy()
        dense = np.zeros(output.shape)
        dense[valids] = 1
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