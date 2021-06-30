#TabNine::sem
from comments import comment
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
OBS = "obs"
VLD = "valids"
TP = "type"
TXT = "text"
CHOICES = "choices"

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

class AgedCardPile:
    def __init__(self):
        self.cards = []
        self.ages = []
    
    def add(self, card):
        pos = bs.bisect(self.ages, card.age)
        self.ages.insert(pos, card.age)
        self.cards.insert(pos, card)
    
    def remove(self, card):
        self.cards.remove(card)
        self.ages.remove(card.age)
    
    def clear(self):
        self.cards = []
        self.ages = []
    
    def sum(self):
        return sum(self.ages)
    
    def high(self, n=1):
        if n == 1:
            return self.ages[-1]
        return dc(self.ages[-n:])
    
    def low(self, n=1):
        if n == 1:
            return self.ages[0]
        return dc(self.ages[:n])
    
    def ageSet(self):
        return set(self.ages)
    
    def public(self):
        return list(map(self.ages.count, range(1, 11)))
    
    def __iter__(self):
        return iter(self.cards)
    
    def __len__(self):
        return len(self.cards)

class Player:
    def __init__(self, name, cds=None, mcds=None):
        self.name = name
        self.cds = cds
        self.mcds = []
        self.cdslen = None
        self.cards = AgedCardPile()
        self.scores = AgedCardPile()
        self.achs = []
        self.icons = {i: 0 for i in ICONS}
        #self.colIcons = pd.DataFrame([[0 for i in range(6)] for j in range(5)], columns=ICONS)
        self.colIcons = {i: [0, 0, 0, 0, 0] for i in ICONS}
        self.board = [deque() for c in range(5)]
        self.splays = [0, 0, 0, 0, 0]
        self.acds = None
        self.op = None
        self.ps = None
        self.masters = [0, 0, 0, 0]
        self.lgr = None
        self.tucked = 0
        self.scored = 0
        self.log = None
        self.agent = None
        self.type = None
        self.args = None
        self.kwargs = None
    
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
                v = self.scores.sum()
                ov = self.op.scores.sum()
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
        pass
    
    def chsX(self, type_, *args, **kwargs):
        self.type = type_
        self.args = args
        self.kwargs = kwargs
        return self.agent.step()
    
    def getInformation(self, requireDict):
        resultDict = {}
        if OBS in requireDict:
            if requireDict[OBS]["obsType"] == "getObs":
                resultDict[OBS] = getObs(self)
                # set chosen and reveal when observation is made by getObs
                if self.type == "ac" or self.type == "at":
                    resultDict[OBS][3] = self.args[0]
                elif self.type == "r":
                    resultDict[OBS][-1] = [self.mcds.index(c) for c in self.args[0]]
            elif requireDict[OBS]["obsType"] == "custom":
                resultDict[OBS] = self.observe()
                # set chosen, etc.
                # need to fix this here
            else:
                resultDict[OBS] = None
        if VLD in requireDict or CHOICES in requireDict:
            if self.type == "main":
                l = [0]
                for age in self.canAchieve():
                    l.append(age)
                for _, c in enumerate(self.board):
                    if c:
                        l.append(c[-1].color + 10)
                for c in self.cards:
                    l.append(self.mcds.index(c) + 15)
            elif self.type == "oc":
                cs = self.args[0]
                ps = self.kwargs["ps"]
                l = []
                for c in cs:
                    l.append(self.mcds.index(c))
                if ps:
                    l.append(self.cdslen)
            elif self.type == "ac" or self.type == "at":
                # set chosen when observation is made by getObs
                if OBS in requireDict and requireDict[OBS]["obsType"] == "getObs":
                    resultDict[OBS][3] = self.args[0]
                else:
                    # need to fix this in other case
                    pass
                l = self.args[1] # valids computed in chsASC or chsAT
            elif self.type == "ot":
                l = []
                for c in self.args[0]:
                    l.append(self.args[1].index(c) + self.args[2])
                if self.kwargs["ps"]:
                    l.append(10)
            elif self.type == "age":
                l = list(range(10))
            elif self.type == "yn":
                l = [0, 1]
            elif self.type == "r":
                l = [0]
            else:
                raise ValueError(f"invalid observation type: {self.type}")
            resultDict[VLD] = l
        if TXT in requireDict:
            text = self.kwargs.get("text", "")
            resultDict[TXT] = text
        if CHOICES in requireDict:
            choices = []
            for n in l:
                if self.type == "main":
                    if n == 0:
                        choices.append("draw a card")
                    elif n < 10:
                        choices.append(f"achieve {n}")
                    elif n < 15:
                        choices.append(f"execute {self.board[n-10][-1]}")
                    else:
                        choices.append(f"meld {self.mcds[n-15]}")
                elif self.type == "oc" or self.type == "ac":
                    if n < self.cdslen:
                        choices.append(self.mcds[n].name)
                    else:
                        choices.append("pass")
                elif self.type == "ot":
                    if n < 5:
                        choices.append(NUMTOCOL[n])
                    else:
                        choices.append("pass")
                elif self.type == "at":
                    if n < 10:
                        choices.append(NUMTOCOL[n % 5])
                    else:
                        choices.append("pass")
                elif self.type == "age":
                    choices.append(str(n + 1))
                elif self.type == "yn":
                    choices.append("yes" if n == 1 else "no")
                elif self.type == "r":
                    choices.append("got it")
                else:
                    raise ValueError(f"invalid observation type: {self.type}")
            resultDict[CHOICES] = choices
        if TP in requireDict:
            resultDict[TP] = self.type
        return resultDict
        
    
    def getMove(self):
        return self.chsX("main")
    
    def chsSC(self, cs, ps=True, text=""):
        cs = list(cs)
        if cs:
            r = self.chsX("oc", cs, ps=ps, text=text)
            return self.mcds[r] if r < self.cdslen else None
        else:
            return None

    def chsC(self, ps=True, text=""):
        return self.chsSC(self.cards, ps=ps, text=text)

    def chsS(self, ps=True, text=""):
        return self.chsSC(self.scores, ps=ps, text=text)

    def chsASC(self, cs, mx=None, ps=True, text=""):
        #obs = self.getObs()

        # get indices of cards as valids
        l = list(map(self.mcds.index, cs))
        if not mx: # max number of cards to choose
            mx = len(l) # you can choose at most len(l) cards
        chs = [] # chosen cards
        if ps:
            l.append(self.cdslen) # add pass action
        for i in range(mx):
            if l:
                #obs[3] = chs # set chosen to observation
                r = self.chsX("ac", chs, l, text=text)
            else:
                return [self.mcds[i] for i in chs]
            if r == self.cdslen: # passed
                return [self.mcds[i] for i in chs]
            chs.append(r)
            l.remove(r)
        return [self.mcds[i] if i < self.cdslen else None for i in chs]

    def chsAC(self, mx=None, ps=True, text=""):
        return self.chsASC(self.cards, mx=mx, ps=ps, text=text)

    def chsAS(self, mx=None, ps=True, text=""):
        return self.chsASC(self.scores, mx=mx, ps=ps, text=text)
    
    def chsST(self, cs, ps=True, op=False, text=""):
        cs = list(cs)
        if op:
            bd = self.op.board
            add = 5
        else:
            bd = self.board
            add = 0
        if cs:
            r = self.chsX("ot", cs, bd, add, ps=ps, text=text)
            return r - add if r < 10 else None
        else:
            return None

    def chsT(self, ps=True, text=""):
        return self.chsST(self.board, ps=ps, text=text)

    def chsAT(self, mx=None, text=""):
        #obs = self.getObs()
        if not mx:
            mx = 5
        cs = []
        chs = [0, 1, 2, 3, 4]
        for _ in range(mx):
            #obs[3] = cs
            c = self.chsX("at", cs, chs, text=text)
            if c is None:
                return cs
            cs.append(c)
            chs.remove(c)
        return cs
    
    def chsA(self):
        r = self.chsX("age", text="You must choose an age")
        return r + 1
    
    def chsYn(self, text=""):
        return self.chsX("yn", text=text)
    
    def _reveal(self, cs):
        self.chsX("r", cs)

    def maySplay(self, col, d, m=False):
        if m:
            c = self.chsST(map(lambda c: len(self.board[c]) >= 2, col))
            if c is not None:
                self.splay(c, d)
        else:
            if len(self.board[col]) >= 2 and not self.isSplay(col, d) and self.chsYn():
                self.splay(col, d)

    def isSplay(self, col, d):
        return self.splays[col] == d
    
    def ckSplay(self, col):
        if len(self.board[col]) < 2 and not self.isSplay(col, 0):
            self.splay(col, 0)

    def splay(self, col, d):
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
        mx = min(self.scores.sum() // 5, max([0] + list(map(lambda c: c[-1].age, filter(lambda c: c, self.board)))))
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
        ags = map(lambda c: c[-1].age if c else 0, self.board)
        age = max(ags)
        c = self.cds.pop(age)
        self.cards.add(c)
        c.cm(self.masters[0])
        self.ilog('%s draws a %d %s', self, c.age, c)
        self.oilog('%s draws a %d', self, c.age)
    
    def draw(self, age, shr=True):
        c = self.cds.pop(age)
        self.cards.add(c)
        c.cm(self.masters[0])
        self.ilog('%s draws a %d %s', self, c.age, c)
        self.oilog('%s draws a %d', self, c.age)
        if shr:
            self.ps.shared[-1] = True
    
    def meld(self, c, shr=True, score=False, frm=None):
        #c = self.cards.pop(idx)
        if score:
            self.scores.remove(c)
        elif frm is not None:
            frm.remove(c)
        else:
            if c not in self.cards:
                self.ailog('Error: list.remove(x): x not in list')
                self.ailog("%s's cards: %s", self, self.cards)
                self.ailog("%s's going to meld %s", self, c)
                self.ailog("%s's cages: %s", self, self.cards.ages)
                self.ailog("executing: %s", self.ps.executing)
                breakpoint()
            self.cards.remove(c)
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
            self.ailog("%s's cages: %s", self, self.cards.ages)
            self.ailog("executing: %s", self.ps.executing)
            breakpoint()
        #c = self.cards.pop(idx)
        self.cards.remove(c)
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
        self.scores.add(c)
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
        self.scores.add(c)
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
        self.cds.appendleft(c)
        c.hide()
        self.ilog('%s returns a %d %s', self, c.age, c)
        self.oilog('%s returns a %d', self, c.age)
        self.ps.shared[-1] = True
    
    def scoreRtrn(self, c):
        self.scores.remove(c)
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
        self.scores.add(c)
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
        self.cards.add(c)
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
        to.add(c)
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
        to.add(c)
        self.op.cIcon(c.color)
        c.cm(self.masterOf(to))
        self.op.ckSpec([1, 2, 4])
        self.ps.shared[-1] = True
    
    def nbtbTransfer(self, frm, c):
        frm.remove(c)
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
        for c in cs1:
            #print(cs1, cs2, ages1, ages2, m1, m2)
            m1.remove(c)
            m2.add(c)
            c.cm(self.masterOf(m2))
        for c in cs2:
            m2.remove(c)
            m1.add(c)
            c.cm(self.masterOf(m1))
        self.ps.shared[-1] = True

    def exchange1(self, c1, c2, cs1, cs2):
        #p1 = self.pOf(cs1)
        #p2 = self.pOf(cs2)
        for c, cs, ocs in ((c1, cs1, cs2), (c2, cs2, cs1)):
            #print(c, cs, ocs, self.ageOf(cs), self.ageOf(ocs))
            cs.remove(c)
            ocs.add(c)
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
    
    def setAgent(self, agent):
        self.agent = agent
        self.agent.getInfo = self.getInformation
    
    def observe(self):
        return Obj(self, notCopies=["op", "ps", "agent", "fh", "lgr", "olgr"], \
            op=self.op.public())
    
    def public(self):
        return Public(self, notCopies=["op", "ps", "agent", "fh", "lgr", "olgr"])

class Obj:
    def __init__(self, obj, notCopies, **kwargs):
        for k, v in vars(obj).items():
            if k in kwargs:
                setattr(self, k, kwargs[k])
            if k in notCopies:
                continue
            setattr(self, k, v)

class Public:
    def __init__(self, player, notCopies):
        for k, v in vars(player).items():
            if k in notCopies:
                continue
            if hasattr(v, "public"):
                v = v.public()
            setattr(self, k, v)

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
        self.history = []
        self.cds = Cds(self.p1.mcds, self.mspecs)
        self.p1.cds = self.cds
        self.p2.cds = self.cds
        self.mp = self.p1
        for p in [self.p1, self.p2]:
            p.tucked = 0
            p.scored = 0
            p.scores.clear()
            p.cards.clear()
            p.achs = []
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

def getObs(mp):
    ps = mp.ps
    op = mp.op
    cards = np.zeros((105,), dtype=np.float32)
    scores = np.zeros((105,), dtype=np.float32)
    cardsi = [i for i in range(105) if mp.mcds[i] in mp.cards.cards]
    scoresi = [i for i in range(105) if mp.mcds[i] in mp.scores.cards]
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
    opcards = op.cards.public()
    opscores = op.scores.public()
    #
    tscore = mp.scores.sum()
    otscore = op.scores.sum()
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
