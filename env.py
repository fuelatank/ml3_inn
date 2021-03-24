#TabNine::sem

from rule import *
from baseObs import BaseObservation
import numpy as np
import tensorflow as tf

class Observation(BaseObservation):
    def __init__(self, data, valids, _type):
        self.data = data
        self.type = _type
        self.valids = valids
    
    def __repr__(self):
        return self.type

class Computer(Player):
    def __init__(self, agent, name='q', cds=None, mcds=None):
        self.agent = agent
        super(Computer, self).__init__(name=name, cds=cds, mcds=mcds)
        self.getObs = lambda: getObs(self.ps)

    def drawStack(self, color): pass

    def getMove(self):
        obs = self.getObs()
        l = [0]
        for age in self.canAchieve():
            l.append(age)
        for i, c in enumerate(self.board):
            if c:
                l.append(c[-1].color + 10)
        for c in self.cards:
            l.append(self.mcds.index(c) + 15)
        return self.agent.step(Observation(obs, l, 'main'))

    def chsSC(self, cs, ps=True):
        obs = self.getObs()
        l = []
        for c in cs:
            l.append(self.mcds.index(c))
        if ps:
            l.append(self.cdslen)
        if not l:
            return None
        #print(l)
        r = self.agent.step(Observation(obs, l, 'oc'))
        #print(r)
        return self.mcds[r] if r < self.cdslen else None

    def chsC(self, ps=True):
        return self.chsSC(self.cards, ps=ps)

    def chsS(self, ps=True):
        return self.chsSC(self.scores, ps=ps)

    def chsASC(self, cs, mx=None, ps=True):
        obs = self.getObs()
        l = list(map(self.mcds.index, cs))
        if not mx:
            mx = len(l)
        chs = []
        if not ps:
            l.append(self.cdslen)
        for i in range(mx):
            if l:
                obs[3] = chs
                r = self.agent.step(Observation(obs, l, 'ac'))
            else:
                return [self.mcds[i] for i in chs]
            #c = self.mcds[r] if r < self.cdslen else None
            if r == self.cdslen:
                return [self.mcds[i] for i in chs]
            chs.append(r)
            if r not in l:
                breakpoint()
            l.remove(r)
        return [self.mcds[i] if i < self.cdslen else None for i in chs]

    def chsAC(self, mx=None, ps=True):
        return self.chsASC(self.cards, mx=mx, ps=ps)

    def chsAS(self, mx=None, ps=True):
        return self.chsASC(self.scores, mx=mx, ps=ps)

    def chsST(self, cs, ps=True, op=False):
        obs = self.getObs()
        l = list(cs)
        if op:
            bd = self.op.board
            add = 5
        else:
            bd = self.board
            add = 0
        r = []
        for c in l:
            r.append(bd.index(c) + add)
        if ps:
            r.append(10)
        if not r:
            return None
        rr = self.agent.step(Observation(obs, r, 'ot'))
        return rr-add if rr < 10 else None

    def chsT(self, ps=True):
        return self.chsST(self.board, ps=ps)

    def chsAT(self, mx=None):
        obs = self.getObs()
        if not mx:
            mx = 5
        cs = []
        chs = [0, 1, 2, 3, 4]
        for _ in range(mx):
            obs[3] = cs
            c = self.agent.step(Observation(obs, chs, 'at'))
            if not c:
                return cs
            cs.append(c)
            if c not in chs:
                breakpoint()
            chs.remove(c)
        return cs

    def chsA(self):
        obs = self.getObs()
        return self.agent.step(Observation(obs, list(range(10)), 'age')) + 1

    def chsYn(self):
        obs = self.getObs()
        return self.agent.step(Observation(obs, [0, 1], 'yn'))

    def _reveal(self, cs):
        obs = self.getObs()
        obs[-1] = [self.mcds.index(c) for c in cs]
        self.agent.step(Observation(obs, [0], 'r'))

'''class Innovation:
    def __init__(self, p1, p2, path):
        self.ps = Players(p1, p2, path=path)
    def reset(self):
        self.ps.reset()
    def step(self, action):
        pass'''
