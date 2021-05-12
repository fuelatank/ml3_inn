#TabNine::sem
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from baseObs import Observation
from obs import stackObsTorch as stackObs
from weights import NPWeights
import trainconfig as tc

rnns = {'lstm': nn.LSTM, 'gru': nn.GRU}
activations = {'relu': torch.relu, 'tanh': torch.tanh}

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
        self.strA = activation
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
        self.main = ChooseModel(rnnInputLength, 120, hidden=128, activation=self.strA)
        self.oc = ChooseModel(rnnInputLength, 106, executing=True, activation=self.strA)
        self.ot = ChooseModel(rnnInputLength, 11, executing=True, activation=self.strA)
        self.ac = ChooseModel(rnnInputLength, 106, executing=True, chosen=True, activation=self.strA)
        self.at = ChooseModel(rnnInputLength, 5, executing=True, chosen=True, activation=self.strA)
        self.yn = ChooseModel(rnnInputLength, 2, executing=True, activation=self.strA)
        self.age = ChooseModel(rnnInputLength, 10, activation=self.strA)
        self.r = ChooseModel(rnnInputLength, 1, hidden=1, activation=self.strA)
    
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

def buildModel(isize, esize, rnnSizes, rnn='lstm'):
    model = BasicModel(isize, esize,  rnnSizes, rnn=rnn)
    return model

def validFilter(logits, denseValids):
    return torch.exp(logits) * denseValids

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
        self.model = buildModel(isize, esize, rnnSizes, rnn=rnn)
        self.optimizer = torch.optim.Adam(self.model.parameters, lr=lr)
        self.gamma = gamma
        self.states = None
        self.target = Target(buildModel(isize, esize, rnnSizes, rnn=rnn))
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
                acts = torch.tensor([s[1] for s in episode], dtype=torch.int32)
                rew = torch.tensor([episode[-1][2]], dtype=torch.float32)
                allObs, allValids = stackObs(episode)
                self._fit(allObs, allValids, acts, rew, double=double)
    
    def fit_maker(self):
        def f(allObs, allValids, acts, rew, double):
            target = self.target
            with torch.no_grad():
                nqs = target.predict(allObs, allValids)[1:]
                if double:
                    cqs = self.predict(allObs, allValids)[1:]
                    bestAct = torch.argmax(cqs, dim=1, keepdims=True)
                    q_next = torch.squeeze(torch.gather(nqs, bestAct))
                else:
                    q_next = torch.max(nqs, dim=1)
                ys = self.gamma * torch.log(q_next)
                ys = torch.cat([ys, rew], dim=0)
            qs, _ = self.model(allObs)
            y_preds = torch.squeeze(torch.gather(qs, torch.unsqueeze(acts, 1)))
            loss = torch.mean((ys - y_preds) ** 2)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return f
    
    def step(self, obs):
        with torch.no_grad():
            r = self._step(obs.data, obs.valids)
        return r
    
    def _step(self, data, valids):
        logits, self.states = self.model({'x': data, 'hs': self.states})
        r = validFilter(logits, torch.unsqueeze(valids, 0))
        return torch.argmax(r)

    def predict(self, data, valids):
        r, _ = self.model({'x': data})
        return r

    def updateTarget(self):
        self.target.set(self.model)

    def reset_states(self):
        self.states = None

class Target:
    def __init__(self, model):
        self.model = model
        #self.rnnLayers = rnnLayers

    def set(self, model):
        self.model.load_state_dict(model.state_dict())

    def predict(self, data, valids):
        r, _ = self.model({'x': data})
        return r

'''qmodel = QModel(300, 20, [128, 128], 1e-4, rnn='gru')
i = tf.random.normal((1, 300))
e = tf.random.categorical(tf.zeros((1, 106)), num_samples=20)
v = tf.cast(tf.random.categorical(tf.constant([[0., 0.]]), num_samples=120), tf.float32)
print(i.shape, e.shape, v)
obs = Observation([i, e, tf.constant([[45]])], v, 'main')
print(qmodel.predict(obs), qmodel.step(obs), qmodel.step(obs), qmodel.step(obs))'''