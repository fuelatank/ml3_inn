#TabNine::sem
from cards import ages, colors, ics, funcs, dems, idxs, msgs, spec_names, conds, cr, lf, lb, ct, fc, cl

import random
import bisect as bs
import pandas as pd
import numpy as np
import logging as lg
from collections import Counter, deque
from copy import copy as cp, deepcopy as dc

MODE = 'w'
COLORS = {'crown': '#FFFF00',
          'leaf': '#00FF00',
          'lightbulb': '#FF00FF',
          'castle': '#7F7F7F',
          'factory': '#FF0000',
          'clock': '#0000FF',
          None: '#000000',
          0: '#0000FF',
          1: '#FF0000',
          2: '#00FF00',
          3: '#FFFF00',
          4: '#FF00FF'}
NUMTOCOL = ['blue', 'red', 'green', 'yellow', 'purple']
ICONS = [cr, lf, lb, ct, fc, cl]
SPLAYICONS = [[], [3], [0, 1], [1, 2, 3]]
SIDES = [('top', 'bottom'), ('right', 'left'), ('left', 'right'), ('bottom', 'top')]
DIRS = [['unsplays', ''], ['splays', ' left'], ['splays', ' right'], ['splays', ' up']]
SPECIALS = ['Monument', 'Empire', 'World', 'Wonder', 'Universe']

class WinError(Exception):
    def __init__(self, *args, **kwargs):
        super(WinError, self).__init__(*args, **kwargs)

class Card:
    def __init__(self, name, age, color, icons, cmd, dem, msg=''):
        self.name = name.capitalize()
        self.age = age
        self.color = color
        self.icons = icons
        self.msg = msg
        self.cmd = cmd
        self.dem = dem
        self.most = self.getMost()
        self.splayIcons = [[self.icons[i] for i in idxs if self.icons[i] is not None] for idxs in SPLAYICONS]
    
    def __repr__(self):
        return self.name
    
    def getMost(self):
        for i in self.icons:
            if self.icons.count(i) >= 2:
                return i
    
    def cm(self, m, row=None, column=None):
        pass
    
    def hide(self):
        pass

class Spec:
    def __init__(self, name, i, cond=None):
        self.i = i
        self.name = name
        self.cond = cond

    def __repr__(self):
        return self.name
    
    def cm(self, m, x=None, y=None):
        pass

mainspecs = list(map(lambda i, name, cond: Spec(name, i, cond=cond), range(5), spec_names, conds))
l = list(zip(ages, colors, ics, funcs, dems, idxs))
maincds = list(map(lambda c: Card(c[3][0].__name__, c[0], c[1], c[2], c[3], c[4], c[5]), l))
#print([[c, c.icons, c.splayIcons] for c in random.sample(maincds, 15)])
mainacds = {}
lst = 0
for a, i in enumerate(range(15, 115, 10), 1):
    mainacds[a] = maincds[lst:i]
    lst = i
del lst

class Cds:
    def __init__(self, cds, specs):
        lst = 0
        self.cds = []
        self.achs = []
        self.specs = cp(specs)
        self.idxes = list(range(5))
        #self.name2spec = {s.name: s for s in self.specs}
        for i in range(15, 115, 10):
            self.cds.append(deque(cds[lst:i]).copy())
            random.shuffle(self.cds[-1])
            if self.cds[-1] and i != 105:
                self.achs.append(self.cds[-1].pop())
            lst = i
        del lst
        self.popIndexes = [0] + list(range(11))
    
    def pop(self, age):
        idx = self.popIndexes[age]
        try:
            if len(self.cds[idx]) == 1:
                #print(self.popIndexes)
                fidx = self.popIndexes.index(idx)
                goal = self.popIndexes[idx+2]
                for i in range(fidx, idx+2):
                    self.popIndexes[i] = goal
                #print(self.popIndexes)
            return self.cds[idx].pop()#, self.knows[1][idx].pop(), self.knows[-1][idx].pop()
        except IndexError:
            raise WinError('score')
    
    def appendleft(self, c, p1=0, p2=0):
        idx = c.age - 1
        if not self.cds[idx]:
            #print(self.popIndexes)
            fidx = self.popIndexes.index(self.popIndexes[c.age])
            for i in range(fidx, c.age+1):
                self.popIndexes[i] = idx
            #print(self.popIndexes)
        self.cds[idx].appendleft(c)

    def rm(self, age):
        for c in self.achs:
            if c.age == age:
                self.achs.remove(c)
                return c

    def getSpec(self, i):
        if i not in self.idxes:
            return None
        else:
            for c in self.specs:
                if c.i == i:
                    self.specs.remove(c)
                    self.idxes.remove(i)
                    return c

class Player:
    def __init__(self, name, cds=None, mcds=None):
        self.name = name
        self.cds = cds
        self.mcds = []
        self.cdslen = None
        self.cards = []
        self.cages = []
        self.scores = []
        self.sages = []
        self.achs = []
        self.icons = {i: 0 for i in ICONS}
        #self.colIcons = pd.DataFrame([[0 for i in range(6)] for j in range(5)], columns=ICONS)
        self.colIcons = {i: [0, 0, 0, 0, 0] for i in ICONS}
        self.board = [deque() for c in range(5)]
        self.splays = [0, 0, 0, 0, 0]
        self.cobs = [set() for i in range(10)]
        self.sobs = [set() for i in range(10)]
        self.acds = None
        self.op = None
        self.ps = None
        self.masters = [0, 0, 0, 0]
        self.lgr = None
        self.tucked = 0
        self.scored = 0
        self.log = None
    
    def __repr__(self):
        return self.name

    def makeMove(self, move):
        try:
            if move == 0:
                self.drawMax()
            elif move < 10:
                self.achieve(move)
            elif move < 15:
                self.x(self.board[move-10][-1])
            else:
                self.meld(self.mcds[move-15], shr=False)
        except WinError as e:
            if str(e) == 'score':
                v = sum(self.sages)
                ov = sum(self.op.sages)
                self.ailog('Someone wants to draw a 11')
                if v > ov:
                    self.ailog('%s wins with more scores: %d-%d', self, v, ov)
                    return self
                if v < ov:
                    self.ailog('%s wins with more scores: %d-%d', self.op, ov, v)
                    return self.op
                self.ailog('The players have the same scores: %d', v)
                if len(self.achs) > len(self.op.achs):
                    self.ailog('%s wins with more achievements!', self)
                    return self
                if len(self.achs) < len(self.op.achs):
                    self.ailog('%s wins with more achievements!', self.op)
                    return self.op
                self.ailog('The game was a tie!')
                return 'tie'
            return e.args[0]
        return None
    
    def win(self, name):
        self.ailog('%s wins with %s!', self, name)
        raise WinError(self)

    def drawStack(self, col):
        raise NotImplementedError

    def chsST(self, cs, ps=True, op=False):
        raise NotImplementedError

    def chsYn(self):
        raise NotImplementedError

    def _reveal(self, cs):
        pass

    def maySplay(self, col, d, m=False):
        if m:
            c = self.chsST(map(lambda c: self.board[c], col))
            if c is not None:
                self.splay(c, d)
        else:
            if self.chsYn():
                self.splay(col, d)

    def isSplay(self, col, d):
        return self.splays[col] == d
    
    def ckSplay(self, col):
        if len(self.board[col]) < 2 and self.isSplay(col, 0):
            self.splay(col, 0)

    def splay(self, col, d):
        if self.splays[col] != d:
            self.splays[col] = d
            self.drawStack(col)
            self.cIcon(col)
            self.ckSpec([1, 2, 3])
            a, b = DIRS[d]
            self.ailog('%s %s his %s cards%s', self, a, NUMTOCOL[col], b)
    
    def getIcon(self, col):
        q = self.board[col]
        icons = {i: 0 for i in ICONS}
        icons[None] = 0
        d = self.splays[col]
        #sis = SPLAYICONS[self.splays[col]]
        for c in q:
            if c == q[-1]:
                for i in c.icons:
                    icons[i] += 1
            else:
                for i in c.splayIcons[d]:
                    icons[i] += 1
        del icons[None]
        return icons
    
    def cIcon(self, col):
        icons = self.getIcon(col)
        for ic, cnt in icons.items():
            self.icons[ic] += cnt - self.colIcons[ic][col]
            self.colIcons[ic][col] = cnt
    
    def canAchieve(self):
        mx = min(sum(self.sages) // 5, max([0] + list(map(lambda c: c[-1].age, filter(lambda c: c, self.board)))))
        r = []
        for a in self.cds.achs:
            if a.age <= mx:
                r.append(a.age)
            else:
                break
        return r

    def ckSpec(self, idxes):
        cs = [self.ps.mspecs[i] for i in idxes]
        for c in cs:
            if c.cond(self):
                self.specAchieve(c.i)

    def drawMax(self):
        for _ in (self.cards, self.cages, self.scores, self.sages):
            #self.adlog(cs)
            pass
        ags = map(lambda c: c[-1].age if c else 0, self.board)
        age = max(ags)
        c = self.cds.pop(age)
        pos = bs.bisect(self.cages, c.age)
        self.cages.insert(pos, c.age)
        self.cards.insert(pos, c)
        c.cm(self.masters[0])
        self.ilog('%s draws a %d %s', self, c.age, c)
        self.oilog('%s draws a %d', self, c.age)
    
    def draw(self, age, shr=True):
        for _ in (self.cards, self.cages, self.scores, self.sages):
            #self.adlog(cs)
            pass
        if list(map(lambda c: c.age, self.cards)) != self.cages:
            raise Exception()
        if list(map(lambda c: c.age, self.scores)) != self.sages:
            raise Exception()
        c = self.cds.pop(age)
        pos = bs.bisect(self.cages, c.age)
        self.cages.insert(pos, c.age)
        self.cards.insert(pos, c)
        c.cm(self.masters[0])
        self.ilog('%s draws a %d %s', self, c.age, c)
        self.oilog('%s draws a %d', self, c.age)
        if shr:
            self.ps.shared[-1] = True
    
    def meld(self, c, shr=True, score=False, frm=None):
        #c = self.cards.pop(idx)
        if score:
            self.scores.remove(c)
            self.sages.remove(c.age)
        elif frm is not None:
            frm.remove(c)
        else:
            if c not in self.cards:
                self.ailog('Error: list.remove(x): x not in list')
                self.ailog("%s's cards: %s", self, self.cards)
                self.ailog("%s's going to meld %s", self, c)
                self.ailog("%s's cages: %s", self, self.cages)
                self.ailog("executing: %s", self.ps.executing)
                breakpoint()
            self.cards.remove(c)
            self.cages.remove(c.age)
        self.board[c.color].append(c)
        self.cIcon(c.color)
        self.drawStack(c.color)
        if not frm:
            self.ckSpec([1, 2, 4])
            self.ailog('%s melds %s', self, c)
        if shr:
            self.ps.shared[-1] = True
    
    def tuck(self, c):
        if c not in self.cards:
            self.ailog('Error: list.remove(x): x not in list')
            self.ailog("%s's cards: %s", self, self.cards)
            self.ailog("%s's going to meld %s", self, c)
            self.ailog("%s's cages: %s", self, self.cages)
            self.ailog("executing: %s", self.ps.executing)
            breakpoint()
        #c = self.cards.pop(idx)
        self.cards.remove(c)
        self.cages.remove(c.age)
        self.board[c.color].appendleft(c)
        self.cIcon(c.color)
        self.drawStack(c.color)
        self.tucked += 1
        self.ckSpec([0, 1, 2, 4])
        self.ailog('%s tucks %s', self, c)
        self.ps.shared[-1] = True
    
    def score(self, c):
        #c = self.cards.pop(idx)
        self.cards.remove(c)
        self.cages.remove(c.age)
        pos = bs.bisect(self.sages, c.age)
        self.sages.insert(pos, c.age)
        self.scores.insert(pos, c)
        c.cm(self.masters[2])
        self.scored += 1
        self.ckSpec([0])
        self.ilog('%s scores a %d %s', self, c.age, c)
        self.oilog('%s scores a %d', self, c.age)
        self.ps.shared[-1] = True
    
    def boardScore(self, col, idx=None, card=None):
        if idx:
            c = self.board[col][idx]
            self.board[col].remove(c)
        elif card:
            c = card
            self.board[col].remove(c)
        else:
            c = self.board[col].pop()
        self.ckSplay(col)
        pos = bs.bisect(self.sages, c.age)
        self.sages.insert(pos, c.age)
        self.scores.insert(pos, c)
        self.cIcon(col)
        self.drawStack(col)
        c.cm(self.masters[2])
        self.scored += 1
        self.ckSpec([0, 1, 2, 4])
        self.ailog('%s scores %s from his board', self, c)
        self.ps.shared[-1] = True
    
    def rtrn(self, c):
        #c = self.cards.pop(idx)
        self.cards.remove(c)
        self.cages.remove(c.age)
        self.cds.appendleft(c)
        c.hide()
        self.ilog('%s returns a %d %s', self, c.age, c)
        self.oilog('%s returns a %d', self, c.age)
        self.ps.shared[-1] = True
    
    def scoreRtrn(self, c):
        self.scores.remove(c)
        self.sages.remove(c.age)
        self.cds.appendleft(c)
        c.hide()
        self.ilog('%s returns a %d %s from his scores', self, c.age, c)
        self.oilog('%s returns a %d from his scores', self, c.age)
        self.ps.shared[-1] = True
    
    def boardRtrn(self, col, card=None):
        if card:
            c = card
            self.board[col].remove(card)
        else:
            c = self.board[col].pop()
        self.ckSplay(col)
        self.cds.appendleft(c)
        self.cIcon(col)
        self.drawStack(col)
        c.hide()
        self.ckSpec([1, 2, 4])
        self.ailog('%s returns a %d %s from his board', self, c.age, c)
        self.ps.shared[-1] = True

    def drawAndMeld(self, age, rtrn=False):
        c = self.cds.pop(age)
        self.board[c.color].append(c)
        self.cIcon(c.color)
        self.drawStack(c.color)
        self.ckSpec([1, 2, 4])
        self.ailog('%s draws and melds a %d %s', self, c.age, c)
        self.ps.shared[-1] = True
        if rtrn:
            return c
    
    def drawAndScore(self, age):
        c = self.cds.pop(age)
        pos = bs.bisect(self.sages, c.age)
        self.sages.insert(pos, c.age)
        self.scores.insert(pos, c)
        c.cm(self.masters[2])
        self.scored += 1
        self.ckSpec([0])
        self.ilog('%s draws and scores a %d %s', self, c.age, c)
        self.oilog('%s draws and scores a %d', self, c.age)
        self.ps.shared[-1] = True
    
    def drawAndTuck(self, age, rtrn=False):
        c = self.cds.pop(age)
        self.board[c.color].appendleft(c)
        self.cIcon(c.color)
        self.drawStack(c.color)
        self.tucked += 1
        self.ckSpec([0, 1, 2, 4])
        self.ailog('%s draws and tucks a %d %s', self, c.age, c)
        self.ps.shared[-1] = True
        if rtrn:
            return c
    
    def drawAndReveal(self, age, shr=True):
        c = self.cds.pop(age)
        pos = bs.bisect(self.cages, c.age)
        self.cages.insert(pos, c.age)
        self.cards.insert(pos, c)
        c.cm(self.masters[0])
        self.ailog('%s draws and reveals a %d %s', self, c.age, c)
        if shr:
            self.ps.shared[-1] = True
        self._reveal([c])
        return c

    def revealAll(self, cs):
        self._reveal(cs)

    def reveal(self, c):
        self._reveal([c])

    def achieve(self, age):
        c = self.cds.rm(age)
        self.achs.append(c)
        c.cm(self.masters[3])
        self.ailog('%s achieve %d', self, age)
        if len(self.achs) == 6:
            self.ailog('%s wins with 6 achievements!', self)
            raise WinError(self)
    
    def specAchieve(self, i):
        #print(i)
        #self.ailog(spec_names[i])
        sc = self.cds.getSpec(i)
        if sc is not None:
            self.achs.append(sc)
            self.ailog('%s achieve %s', self, sc.name)
            #sc.cm(self.masters[3])
            if len(self.achs) == 6:
                self.ailog('%s wins with 6 achievements!', self)
                raise WinError(self)

    def xSharedDogma(self, col):
        c = self.board[col][-1]
        self.ps.executing.append(c)
        self.ps.shared.append(False)
        self.ailog('%s execute %s for himself only', self, c)
        for dem, cmd in zip(c.dem, c.cmd):
            if not dem:
                cmd(self)
        shr = self.ps.shared.pop()
        if shr:
            self.ps.shared[-1] = True
        self.ps.executing.pop()
        self.ps.dones.discard(c.name)
    
    def x(self, c):
        self.ps.executing.append(c)
        self.ps.shared.append(False)
        self.ailog('%s execute %s', self, c)
        #self.adlog("%s's icons: %s", self, self.icons)
        #self.adlog("%s's icons: %s", self.op, self.op.icons)
        self.ailog('%s has %d %s', self, self.icons[c.most], c.most)
        self.ailog('%s has %d %s', self.op, self.op.icons[c.most], c.most)
        gt = self.icons[c.most] > self.op.icons[c.most]
        shared = False
        for dem, cmd in zip(c.dem, c.cmd):
            if gt:
                cmd(self)
            elif not dem:
                self.ps.shared[-1] = False
                cmd(self.op)
                shr = self.ps.shared[-1]
                cmd(self)
                if shr:
                    shared = True
        if shared:
            self.ailog('%s must draw a card because of sharing', self)
            self.drawMax()
        self.ailog('All executing of %s finished', c)
        self.ps.shared.pop()
        self.ps.executing.pop()
        self.ps.dones.clear()
        self.ps.executedcount[c.name] += 1
        if len(list(self.ps.executedcount)) == 105 and self.ps.executedcount.most_common()[-1][1] >= 50:
            pass#raise Exception()
    
    def ageOf(self, cs):
        if cs is self.cards:
            return self.cages
        elif cs is self.scores:
            return self.sages
        else:
            return self.op.ageOf(cs)
    
    def masterOf(self, cs):
        if cs is self.cards:
            return self.masters[0]
        elif cs is self.scores:
            return self.masters[2]
        else:
            return self.op.masterOf(cs)

    def pOf(self, cs):
        if cs is self.cards or cs is self.scores:
            return self
        else:
            return self.op

    def nameOf(self, cs):
        if cs is self.cards:
            return 'hand'
        elif cs is self.scores:
            return 'score'
        else:
            return self.op.nameOf(cs)

    def nbTransfer(self, frm, to, c, s=False):
        frm.remove(c)
        self.op.ageOf(frm).remove(c.age)
        ages = self.ageOf(to)
        pos = bs.bisect(ages, c.age)
        ages.insert(pos, c.age)
        to.insert(pos, c)
        c.cm(self.masterOf(to))
        self.ps.shared[-1] = True
    
    def btbTransfer(self, col):
        c = self.op.board[col].pop()
        self.op.ckSplay(col)
        self.board[col].append(c)
        self.cIcon(col)
        self.op.cIcon(col)
        c.cm(self.masters[1])
        self.ps.mp.ckSpec([1, 2, 4])
        self.ps.shared[-1] = True
    
    def btnbTransfer(self, to, col):
        c = self.op.board[col].pop()
        self.op.ckSplay(col)
        ages = self.ageOf(to)
        pos = bs.bisect(ages, c.age)
        ages.insert(pos, c.age)
        to.insert(pos, c)
        self.op.cIcon(c.color)
        c.cm(self.masterOf(to))
        self.op.ckSpec([1, 2, 4])
        self.ps.shared[-1] = True
    
    def nbtbTransfer(self, frm, c):
        frm.remove(c)
        self.op.ageOf(frm).remove(c.age)
        self.board[c.color].append(c)
        self.cIcon(c.color)
        c.cm(self.masters[1])
        self.ckSpec([1, 2, 4])
        self.ps.shared[-1] = True

    def exchange(self, m1, m2, cs1, cs2):
        cs1 = cp(cs1)
        cs2 = cp(cs2)
        #p1 = self.pOf(m1)
        #p2 = self.pOf(m2)
        ages1 = self.ageOf(m1)
        ages2 = self.ageOf(m2)
        for c in cs1:
            #print(cs1, cs2, ages1, ages2, m1, m2)
            m1.remove(c)
            #ages = self.ageOf(m2)
            ages1.remove(c.age)
            pos = bs.bisect(ages2, c.age)
            ages2.insert(pos, c.age)
            m2.insert(pos, c)
            c.cm(self.masterOf(m2))
        for c in cs2:
            m2.remove(c)
            #ages = self.ageOf(m1)
            ages2.remove(c.age)
            pos = bs.bisect(ages1, c.age)
            ages1.insert(pos, c.age)
            m1.insert(pos, c)
            c.cm(self.masterOf(m1))
        self.ps.shared[-1] = True

    def exchange1(self, c1, c2, cs1, cs2):
        #p1 = self.pOf(cs1)
        #p2 = self.pOf(cs2)
        for c, cs, ocs in ((c1, cs1, cs2), (c2, cs2, cs1)):
            #print(c, cs, ocs, self.ageOf(cs), self.ageOf(ocs))
            cs.remove(c)
            self.ageOf(cs).remove(c.age)
            ages = self.ageOf(ocs)
            pos = bs.bisect(ages, c.age)
            ages.insert(pos, c.age)
            ocs.insert(pos, c)
            c.cm(self.masterOf(ocs))
        self.ps.shared[-1] = True

    def ilog(self, msg, *args):
        if self.log:
            self.lgr.info(msg, *args)

    def dlog(self, msg, *args):
        pass#self.lgr.debug(msg, *args)

    def oilog(self, msg, *args):
        pass#self.olgr.info(msg, *args)

    def odlog(self, msg, *args):
        pass#self.olgr.debug(msg, *args)

    def ailog(self, msg, *args):
        if self.log:
            self.lgr.info(msg, *args)
        #self.olgr.info(msg, *args)

    def adlog(self, msg, *args):
        pass#self.lgr.debug(msg, *args)
        #self.olgr.debug(msg, *args)

    def logobs(self):
        pass

    def logHand(self):
        self.ailog("%s's cards: %s" % (self, self.cards))

class Players:
    def __init__(self, p1, p2, cds=maincds, specs=mainspecs, acds=mainacds, path=''):
        self.p1 = p1
        self.p2 = p2
        self.p1.op = self.p2
        self.p2.op = self.p1
        self.p1.ps = self
        self.p2.ps = self
        self.ps = {1: self.p1, -1: self.p2}
        self.mp = self.p1
        self.cds = Cds(cds, specs)
        self.p1.cds = self.cds
        self.p2.cds = self.cds
        self.p1.mcds = cds
        self.p2.mcds = cds
        self.p1.cdslen = len(self.p1.mcds)
        self.p2.cdslen = len(self.p2.mcds)
        self.p1.acds = acds
        self.p2.acds = acds
        self.mspecs = specs
        self.history = []
        self.p1.lgr = lg.getLogger('p1')
        self.p2.lgr = lg.getLogger('p2')
        self.p1.olgr = lg.getLogger('op1')
        self.p2.olgr = lg.getLogger('op2')
        self.p1.lgr.setLevel(lg.DEBUG)
        self.p2.lgr.setLevel(lg.DEBUG)
        self.p1.olgr.setLevel(lg.DEBUG)
        self.p2.olgr.setLevel(lg.DEBUG)
        self.p1.fh = lg.FileHandler(path+'p1.log', MODE)
        self.p2.fh = lg.FileHandler(path+'p2.log', MODE)
        self.fh = lg.FileHandler(path+'main.log', MODE)
        self.ch = lg.StreamHandler()
        self.p1.fh.setLevel(lg.DEBUG)
        self.p2.fh.setLevel(lg.DEBUG)
        self.fh.setLevel(lg.DEBUG)
        self.ch.setLevel(lg.WARNING)
        t_format = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        nt_format = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.p1.fh.setFormatter(nt_format)
        self.p2.fh.setFormatter(nt_format)
        self.fh.setFormatter(t_format)
        self.ch.setFormatter(nt_format)
        self.p1.lgr.addHandler(self.p1.fh)
        self.p1.lgr.addHandler(self.fh)
        self.p1.olgr.addHandler(self.p2.fh)
        #self.p1.olgr.addHandler(self.ch)
        self.p2.lgr.addHandler(self.p2.fh)
        self.p2.lgr.addHandler(self.fh)
        self.p2.olgr.addHandler(self.p1.fh)
        #self.p2.olgr.addHandler(self.ch)
        self.dones = set()
        self.specs = {}
        self.executing = []
        self.shared = []
        self.win = None
        self.turn = 1
        self.action = 1
        self.count = 0
        self.executedcount = Counter()
        self.init()
    
    def init(self):
        self.p1.draw(1, shr=False)
        self.p1.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        c1 = self.p1.chsC(ps=False)
        c2 = self.p2.chsC(ps=False)
        self.p1.meld(c1, shr=False)
        self.p2.meld(c2, shr=False)
        del c1, c2
    
    def oneStep(self):
        move = self.mp.getMove()
        done = self.mp.makeMove(move)
        self.count += 1
        if done:
            return done
        if self.action == 1:
            self.action = 0
            self.turn = -self.turn
            self.p1.tucked = 0
            self.p1.scored = 0
            self.p2.tucked = 0
            self.p2.scored = 0
            self.changeTurn()
        else:
            self.action = 1
        return None

    def changeTurn(self):
        self.mp = self.mp.op

    def logall(self):
        self.p1.logHand()
        self.p1.logSA()
        self.p1.logBoard()
        self.p2.logBoard()
        self.p2.logSA()
        self.p2.logHand()

    def reset(self):
        self.turn = 1
        self.action = 1
        self.count = 0
        self.cds = Cds(self.p1.mcds, self.mspecs)
        self.p1.cds = self.cds
        self.p2.cds = self.cds
        self.mp = self.p1
        for p in [self.p1, self.p2]:
            p.tucked = 0
            p.scored = 0
            p.scores = []
            p.cards = []
            p.achs = []
            p.cages = []
            p.sages = []
            p.board = [deque() for i in range(5)]
            p.splays = [0, 0, 0, 0, 0]
            p.icons = {i: 0 for i in ICONS}
            p.colIcons = {i: [0, 0, 0, 0, 0] for i in ICONS}
        self.init()

    def run(self, maxStep=500):
        for _ in range(maxStep):
            r = self.oneStep()
            if r:
                return r
        return 'tie'

class Game:
    def __init__(self, cds=maincds, specs=mainspecs, acds=mainacds, path=''):
        self.p1 = None
        self.p2 = None
        self.cds = Cds(cds, specs)
        self.mcds = cds
        self.acds = acds
        self.mspecs = specs
        self.history = []
        self.p1lgr = lg.getLogger('p1')
        self.p2lgr = lg.getLogger('p2')
        self.p1olgr = lg.getLogger('op1')
        self.p2olgr = lg.getLogger('op2')
        self.p1lgr.setLevel(lg.DEBUG)
        self.p2lgr.setLevel(lg.DEBUG)
        self.p1olgr.setLevel(lg.DEBUG)
        self.p2olgr.setLevel(lg.DEBUG)
        self.p1fh = lg.FileHandler(path+'p1.log', MODE)
        self.p2fh = lg.FileHandler(path+'p2.log', MODE)
        self.fh = lg.FileHandler(path+'main.log', MODE)
        self.ch = lg.StreamHandler()
        self.p1fh.setLevel(lg.DEBUG)
        self.p2fh.setLevel(lg.DEBUG)
        self.fh.setLevel(lg.DEBUG)
        self.ch.setLevel(lg.INFO)
        t_format = lg.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        nt_format = lg.Formatter('%(name)s - %(levelname)s - %(message)s')
        self.p1fh.setFormatter(nt_format)
        self.p2fh.setFormatter(nt_format)
        self.fh.setFormatter(t_format)
        self.ch.setFormatter(nt_format)
        self.p1lgr.addHandler(self.p1fh)
        self.p1lgr.addHandler(self.fh)
        self.p1olgr.addHandler(self.p2fh)
        #self.p1olgr.addHandler(self.ch)
        self.p2lgr.addHandler(self.p2fh)
        self.p2lgr.addHandler(self.fh)
        self.p2olgr.addHandler(self.p1fh)
        #self.p2olgr.addHandler(self.ch)
        self.dones = set()
        self.specs = {}
        self.executing = []
        self.shared = []
        self.win = None
        self.turn = 1
        self.action = 1
        self.count = 0
        self.executedcount = Counter()
    
    def init(self):
        self.p1.draw(1, shr=False)
        self.p1.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        c1 = self.p1.chsC(ps=False)
        c2 = self.p2.chsC(ps=False)
        self.p1.meld(c1, shr=False)
        self.p2.meld(c2, shr=False)
        del c1, c2
    
    def oneStep(self):
        move = self.mp.getMove()
        done = self.mp.makeMove(move)
        self.count += 1
        if done:
            return done
        if self.action == 1:
            self.action = 0
            self.turn = -self.turn
            self.p1.tucked = 0
            self.p1.scored = 0
            self.p2.tucked = 0
            self.p2.scored = 0
            self.changeTurn()
        else:
            self.action = 1
        return None

    def changeTurn(self):
        self.mp = self.mp.op

    def logall(self):
        self.p1.logHand()
        self.p1.logSA()
        self.p1.logBoard()
        self.p2.logBoard()
        self.p2.logSA()
        self.p2.logHand()

    def reset(self):
        self.dones = set()
        self.specs = {}
        self.executing = []
        self.shared = []
        self.win = None
        self.turn = 1
        self.action = 1
        self.count = 0
        self.cds = Cds(self.p1.mcds, self.mspecs)
        self.p1.cds = self.cds
        self.p2.cds = self.cds
        self.mp = self.p1
        for p in [self.p1, self.p2]:
            p.tucked = 0
            p.scored = 0
            p.scores = []
            p.cards = []
            p.achs = []
            p.cages = []
            p.sages = []
            p.board = [deque() for i in range(5)]
            p.splays = [0, 0, 0, 0, 0]
            p.icons = {i: 0 for i in ICONS}
            p.colIcons = {i: [0, 0, 0, 0, 0] for i in ICONS}
        #self.init()

    def run(self, p1, p2, maxStep=500, log=False):
        self.p1 = p1
        self.p2 = p2
        self.p1.op = self.p2
        self.p2.op = self.p1
        self.p1.ps = self
        self.p2.ps = self
        self.ps = {1: self.p1, -1: self.p2}
        self.mp = self.p1
        self.p1.mcds = self.mcds
        self.p2.mcds = self.mcds
        self.p1.cdslen = len(self.p1.mcds)
        self.p2.cdslen = len(self.p2.mcds)
        self.p1.acds = self.acds
        self.p2.acds = self.acds
        self.history = []
        self.p1.lgr = self.p1lgr
        self.p2.lgr = self.p2lgr
        self.p1.olgr = self.p1olgr
        self.p2.olgr = self.p2olgr
        self.p1.fh = self.p1fh
        self.p2.fh = self.p2fh
        self.p1.log = log
        self.p2.log = log
        self.reset()
        self.init()
        for _ in range(maxStep):
            r = self.oneStep()
            if r:
                return r
        return 'tie'

def getObs(ps):
    mp = ps.mp
    op = ps.mp.op
    cards = np.zeros((105,), dtype=np.float32)
    scores = np.zeros((105,), dtype=np.float32)
    cardsi = [i for i in range(105) if mp.mcds[i] in mp.cards]
    scoresi = [i for i in range(105) if mp.mcds[i] in mp.scores]
    cards[cardsi] = 1
    scores[scoresi] = 1
    board = [list(map(lambda c: mp.mcds.index(c), s)) for s in mp.board + op.board]
    zb = np.zeros((22, 10), dtype=np.int8)
    zb.fill(105)
    splays = mp.splays + op.splays
    for i, l in enumerate(board):
        if l:
            zb[:len(l), i] = l
            zb[-1, i] = l[-1] 
    achs = np.zeros((2, 14), dtype=np.float32)
    for i, a in enumerate([mp.achs, op.achs]):
        for c in a:
            if c in mp.mcds:
                achs[i][c.age-1] = 1
            else:
                achs[i][c.i+9] = 1
    opcards = [op.cages.count(i) for i in range(1, 11)]
    opscores = [op.sages.count(i) for i in range(1, 11)]
    #
    tscore = sum(mp.sages)
    otscore = sum(op.sages)
    age = max(map(lambda x: x[-1].age if x else 0, mp.board))
    oage = max(map(lambda x: x[-1].age if x else 0, op.board))
    icons = [mp.icons[i] for i in mp.icons]
    oicons = [op.icons[i] for i in op.icons]
    bls = np.array(list(map(len, mp.board+op.board)))
    #
    act = ps.action
    #print(cards + scores + [act, tscore/20, otscore/20, age/5, oage/5])
    if ps.executing:
        executing = mp.mcds.index(ps.executing[-1])
    else:
        executing = 105
    return [np.concatenate(
        (np.array([act, ps.count/200, tscore/20, otscore/20, age/5, oage/5], dtype=np.float32),
            np.array(opcards + opscores + icons + oicons + splays, dtype=np.float32)/5.0,
            bls/5.0, cards, scores,
            achs), axis=None), zb.reshape(-1), np.array([executing]), [], []]#), axis=None)
