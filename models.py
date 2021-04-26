#TabNine::sem
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from baseObs import Observation
from obs import stackObs
import trainconfig as tc

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
lstm = rnn(layers.LSTM)
gru = rnn(layers.GRU)

def simpleModel(x):
    x = layers.Dense(128, activation=tc.ACTIVATION)(x)
    x = layers.Dense(128, activation=tc.ACTIVATION)(x)
    o = layers.Dense(120)(x)
    return o

def chooseOneCard(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(106)(x)
    return o

def chooseOneColor(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(11)(x) # opponent's board
    return o

def chooseYn(x, c):
    c = layers.Embedding(106, 4)(c)
    c = layers.Flatten()(c)
    x = layers.Concatenate()([x, c])
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(2)(x)
    return o

def chooseAnyCard(x, e, c):
    e = layers.Embedding(106, 4)(e)
    e = layers.Flatten()(e)
    c = layers.Dense(16, activation=tc.ACTIVATION)(c)
    x = layers.Concatenate()([x, e, c])
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(106)(x)
    return o

def chooseAnyColor(x, e, c):
    e = layers.Embedding(106, 4)(e)
    e = layers.Flatten()(e)
    c = layers.Dense(16, activation=tc.ACTIVATION)(c)
    x = layers.Concatenate()([x, e, c])
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(5)(x) # can't pass
    return o

def chooseAge(x):
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    x = layers.Dense(32, activation=tc.ACTIVATION)(x)
    o = layers.Dense(10)(x)
    return o

def reveal(x):
    o = layers.Dense(1)(x)
    return o

rnns = {'lstm': lstm, 'gru': gru}

def buildModel(isize, esize, rnnSizes, rnn='lstm', training=False):
    if training:
        input1 = keras.Input(shape=(isize,))
        input2 = keras.Input(shape=(esize,))
        executingInput = keras.Input(shape=(1,))
        chosenInput = keras.Input(shape=(105,))
        revealInput = keras.Input(shape=(105,))
    else:
        input1 = keras.Input(batch_shape=(1, isize))
        input2 = keras.Input(batch_shape=(1, esize))
        executingInput = keras.Input(batch_shape=(1, 1))
        chosenInput = keras.Input(batch_shape=(1, 105))
        revealInput = keras.Input(batch_shape=(1, 105))
    rnnfn = rnns[rnn] if rnn in rnns else None
    feature = rnnfn(input1, input2, executingInput, revealInput, rnnSizes, training=training)
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
    #model.summary()
    return model, rnnLayers, indexes

def modelTFFunction(model):
    #ispec = tf.TensorSpec(shape=model.input_shape[0], dtype=tf.float32)
    #espec = tf.TensorSpec(shape=model.input_shape[1], dtype=tf.int32)
    #specs = [ispec, espec] + \
    specs = [tf.TensorSpec(shape=s, dtype=tf.float32) for s in model.input_shape]
    @tf.function(input_signature=(specs, tf.TensorSpec(shape=[None, 361], dtype=tf.float32)))
    def pred(data, denseValids):
        #print(model.name)
        r = model(data)#[0]
        #tf.print(tf.shape(r))
        return tf.exp(r) * denseValids
    return pred

def modelTFFunctionAction(model):
    @tf.function
    def pred(data, act):
        return model(data)[0][act]
    return pred

class PolicyModel:
    def __init__(self, isize, esize, rnnSizes, lr, rnn='lstm'):
        self.model, self.rnnLayers, self.indexes = buildModel(isize, esize, rnnSizes, rnn=rnn)
        self.stepfunc = modelTFFunction(self.model)
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
    
    def step(self, obs):
        probs = self._step(obs.data, obs.valids).numpy()
        action = np.random.choice(361, p=probs)
        return action
    
    @tf.function
    def _step(self, data, valids):
        r = self.stepfunc(data, tf.expand_dims(valids, axis=0))[0]
        probs = r / tf.reduce_sum(r)
        return probs

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