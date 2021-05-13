#TabNine::sem
from models import QModel
from env import Observation
from baseObs import BaseTFObservation
import trainconfig as tc

import random
from collections import deque
import tensorflow as tf
import numpy as np

from obs import TFObservation

class QBuffer:
    def __init__(self, size=tc.BUFFER_SIZE, gamma=tc.GAMMA):
        self.data = []
        self.newEpisode = []
        self.maxsize = size
        self.gamma = gamma
        self.pos = 0
        self.finish_path = self.empty_finish_path

    def store(self, obs, action):
        if self.newEpisode:
            self.newEpisode[-1][3] = obs
        #self.newEpisode[-1]['next_valids'] = valids
        self.newEpisode.append([obs, action, 0, None, True])

    def empty_finish_path(self, r):
        self.newEpisode[-1][4] = 0
        self.newEpisode[-1][2] = r
        #self.newEpisode[-1]['next_obs'] = self.newEpisode[-1]['obs']
        #self.newEpisode[-1]['next_valids'] = self.newEpisode[-1]['valids']
        self.data.append(self.newEpisode)
        self.newEpisode = []
        if len(self.data) == self.maxsize:
            self.finish_path = self.full_finish_path

    def full_finish_path(self, r):
        self.newEpisode[-1][4] = 0
        self.newEpisode[-1][2] = r
        self.data[self.pos] = self.newEpisode
        self.newEpisode = []
        self.pos += 1
        if self.pos == self.maxsize:
            self.pos = 0

    def calculateRewards(self):
        last = 0
        for l in self.newEpisode[::-1]:
            l[2] += last * self.gamma
            last = l[2]

    def get(self, num_episodes=tc.BUFFER_SAMPLE_SIZE):
        return random.sample(self.data, k=min(num_episodes, len(self.data)))

class BaseAgent:
    def __init__(self):
        pass

    def step(self, obs):
        raise NotImplementedError
    
    def finish_path(self, r=None):
        pass

class QAgent(BaseAgent):
    def __init__(self, model, buf, n_update_target=tc.UPDATE_TARGET_FREQ):
        self.model = model
        #self.target = tf.keras.models.clone_model(self.model)
        self.buf = buf
        #self.finish_path = self.buf.finish_path
        self.n_update_target = n_update_target
        self.count = 0
        self.trainable = False
        self.lastObs = None
        self.lastAction = None
        self.collecting = True

    def finish_path(self, r=None):
        if self.collecting:
            self.buf.finish_path(r)
        self.model.reset_states()

    def step(self, obs, epsilon=0.2):
        if random.random() < epsilon:
            action = random.choice(obs.valids)
            tfobs = TFObservation(obs)
            r = action
            action += tfobs.typeIndex
        else:
            tfobs = TFObservation(obs)
            action = int(self.model.step(tfobs))
            r = action - tfobs.typeIndex
        if self.collecting:
            self.buf.store(tfobs, action)
        self.lastObs = obs
        self.lastAction = action
        return r

    def train(self):
        data = self.buf.get()
        try:
            self.model.fit(data)
        except:
            raise
        self.count += 1
        if self.count == self.n_update_target:
            self.count = 0
            self.model.updateTarget()

class ModelAgent(BaseAgent):
    def __init__(self, model):
        super(ModelAgent, self).__init__()
        self.model = model
    
    def step(self, obs):
        tfobs = TFObservation(obs)
        action = int(self.model.step(tfobs))
        r = action - tfobs.typeIndex
        return r
    
    def set_weights(self, w):
        self.model.set_weights(w)

class RandomAgent:
    def __init__(self):
        self.trainable = True

    def step(self, obs):
        return random.choice(obs.valids)

if __name__ == '__main__':
    qmodel = QModel(300, 20, [128, 128], 1e-4, rnn='gru')
    i = np.random.normal(size=300)
    e = np.random.choice(106, 20)
    v = np.random.choice(2, 106)
    print('i:', i.shape, 'e', e.shape, 'v', v)
    obs = Observation([i, e, np.array([45]), [0, 5, 67]], v, 'ac')
    obs = TFObservation(obs)
    print('qmodel.predict(obs)xxx', qmodel.step(obs), qmodel.step(obs), 'ls', qmodel.step(obs))
    qbuffer = QBuffer(10)
    for i in range(2):
        for j in range(random.randint(50, 100)):
            obs = Observation([
                np.random.normal(size=300),
                np.random.choice(106, 20),
                np.array([45]), [0, 5, 67]],
                np.random.choice(2, 106),
                'ac')
            qbuffer.store(obs, random.randint(0, 106-1))
        qbuffer.finish_path(random.randint(-1, 1))
        #print(qbuffer.data)
        if i % 2 == 0:
            qmodel.fit(qbuffer.get(2), qmodel, loops=2)