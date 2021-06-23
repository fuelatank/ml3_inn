
import itertools as _it
from copy import deepcopy as dc
from typing import Text

from comments import *

cr = 'crown'
lf = 'leaf'
lb = 'lightbulb'
ct = 'castle'
fc = 'factory'
cl = 'clock'
e = None
t = True
f = False

def agriculture(p):
    c = p.chsC(text=comment(RETURN))
    if c:
        p.rtrn(c)
        p.drawAndScore(c.age+1)

def ai(p):
    p.drawAndScore(10)

def ai2(p):
    tops = list(map(lambda c: c[-1].name, filter(lambda c: c, p.board))) + list(map(lambda c: c[-1].name, filter(lambda c: c, p.op.board)))
    print(tops)
    if 'Robotics' in tops and 'Software' in tops:
        ts = sum(p.sages)
        ots = sum(p.op.sages)
        if ts > ots:
            p.op.win('ai')
        elif ts < ots:
            p.win('ai')

def alchemy(p):
    cs = []
    red = False
    for i in range(p.icons[ct] // 3):
        c = p.drawAndReveal(4)
        cs.append(c)
        if c.color == 1:
            red = True
    if red:
        for i in range(len(p.cards)):
            c = p.chsC(ps=False, text=comment(RETURN, may=False, num=ALL))
            p.rtrn(c)

def alchemy2(p):
    c = p.chsC(ps=False, text=comment(MELD, may=False))
    if c:
        p.meld(c)
        c = p.chsC(ps=False, text=comment(SCORE, may=False))
        if c:
            p.score(c)

def anatomy(p):
    c = p.op.chsS(ps=False, text=comment(RETURN, may=False, from_=SCOREPILE))
    if c:
        p.op.scoreRtrn(c)
        col = p.op.chsST(filter(lambda cd: cd and cd[-1].age == c.age, p.op.board), \
            ps=False, text=comment(RETURN, may=False, age=c.age, from_=BOARD))
        if col:
            p.op.boardRtrn(col)

def antibiotics(p):
    cs = p.chsAC(mx=3, text=comment(RETURN, num=3))
    for c in cs:
        p.rtrn(c)
    for i in range(len(cs)):
        p.draw(8)
        p.draw(8)

def archery(p):
    p.op.draw(1)
    c = p.op.chsSC(filter(lambda c: c.age == p.op.cages[-1], p.op.cards), \
        ps=False, text=comment(TRANSFER, may=False, num=HIGHEST, to=his(HAND)))
    p.nbTransfer(p.op.cards, p.cards, c)

def astronomy(p):
    c = p.drawAndReveal(6)
    while c.color == 0 or c.color == 2:
        p.meld(c)
        c = p.drawAndReveal(6)

def astronomy2(p):
    if not list(filter(lambda c: c and c[-1].age < 6 and c[-1].color < 4, p.board)):
        p.specAchieve(4)

def atomicTheory(p):
    p.maySplay(0, 2)

def atomicTheory2(p):
    p.drawAndMeld(7)

def banking(p):
    col = p.op.chsST(filter(lambda c: c and fc in c[-1].icons and c[-1].color != 2, p.op.board), \
        ps=False, text=comment(TRANSFER, may=False, color=non(GREEN), icon=fc, from_=BOARD, to=his(BOARD)))
    if col is not None:
        p.btbTransfer(col)
        p.op.drawAndScore(5)

def banking2(p):
    p.maySplay(2, 2)

def bicycle(p):
    if p.chsYn(text=comment(EXCHANGE, num=ALL, custom="in your hand and score pile", from_=None)):
        p.exchange(p.scores, p.cards, p.scores, p.cards)

def bioengineering(p):
    col = p.chsST(filter(lambda c: c and lf in c[-1].icons, p.op.board), \
        ps=False, op=True, text=comment(TRANSFER, may=False, icon=lf, custom="from his board to your score pile"))
    if col is not None:
        p.btnbTransfer(p.scores, col)

def bioengineering2(p):
    l = p.icons[lf]
    ol = p.op.icons[lf]
    if l < 3 or ol < 3:
        if l > ol:
            p.win('Bioengineering')
        elif ol > l:
            p.op.win('Bioengineering')

def calendar(p):
    if len(p.scores) > len(p.cards):
        p.draw(3)
        p.draw(3)

def canelBuilding(p):
    cs = list(filter(lambda c: c.age == p.cages[-1], p.cards))
    ss = list(filter(lambda c: c.age == p.sages[-1], p.scores))
    p.exchange(p.scores, p.cards, ss, cs)

def canning(p):
    if p.chsYn(text=comment(TUCK, age=6, drawAnd=True)):
        p.drawAndTuck(6)
        cols = filter(lambda i: p.board[i] and fc not in p.board[i][-1].icons, range(5))
        for col in cols:
            p.boardScore(col)

def canning2(p):
    p.maySplay(3, 2)

def chemistry(p):
    p.maySplay(0, 2)

def chemistry2(p):
    p.drawAndScore(max(map(lambda c: c[-1].age if c else 0, p.board)) + 1)
    c = p.chsS(ps=False, text=comment(RETURN, may=False, from_=SCOREPILE))
    p.scoreRtrn(c)

def cityStates(p):
    if p.op.icons[ct] >= 4:
        col = p.op.chsST(filter(lambda c: c and ct in c[-1].icons, p.op.board), \
            ps=False, text=comment(TRANSFER, may=False, icon=ct, from_=BOARD, to=his(BOARD)))
        if col:
            p.btbTransfer(col)
            p.op.draw(1)

def classfication(p):
    c = p.chsC(ps=False)
    if c:
        p.reveal(c)
        cs = filter(lambda cd: c.color == cd.color, p.op.cards)
        for cd in cs:
            p.nbTransfer(p.op.cards, p.cards, cd)
        p.op.revealAll(p.op.cards)
        cs = list(filter(lambda cd: c.color == cd.color, p.cards))
        for i in range(len(cs)):
            c = p.chsSC(cs, ps=False, text=comment(MELD, may=False, color=COLORS[cd.color]))
            p.meld(c)
            cs.remove(c)

def clothing(p):
    c = p.chsSC(filter(lambda c: not p.board[c.color], p.cards), \
        ps=False, text=comment(MELD, may=False, custom="of different color from any card on your board"))
    if c:
        p.meld(c)

def clothing2(p):
    for i in filter(lambda c: len(p.board[c]) > 0 and len(p.op.board[c]) == 0, range(5)):
        p.drawAndScore(1)

def coal(p):
    p.drawAndTuck(5)

def coal2(p):
    p.maySplay(1, 2)

def coal3(p):
    col = p.chsST(filter(lambda c: c, p.board), text=comment(SCORE, from_=BOARD))
    if col is not None:
        p.boardScore(col)
        if p.board[col]:
            p.boardScore(col)

def codeOfLaws(p):
    c = p.chsSC(filter(lambda c: p.board[c.color], p.cards), \
        text=comment(TUCK, custom="of the same color as any card on your board"))
    if c:
        p.tuck(c)
        if not p.isSplay(c.color, 1):
            p.splay(c.color, 1)

def collaboration(p):
    cs = [p.op.drawAndReveal(9), p.op.drawAndReveal(9)]
    c = p.chsSC(cs, ps=False, text=comment(TRANSFER, may=False, custom="which he revealed from his hand", to="your board"))
    p.nbtbTransfer(p.op.cards, c)
    cs.remove(c)
    p.op.meld(cs[0])

def collaboration2(p):
    if len(p.board[2]) >= 10:
        p.win('Collaboration')

def colonialism(p):
    c = p.drawAndTuck(3, rtrn=True)
    while cr in c.icons:
        c = p.drawAndTuck(3, rtrn=True)

def combustion(p):
    cs = p.op.chsAS(mx=2, ps=False, \
        text=comment(TRANSFER, may=False, num=2, from_=SCOREPILE, to=his(SCOREPILE)))
    for c in cs:
        p.nbTransfer(p.op.scores, p.scores, c)

def compass(p):
    col = p.op.chsST(filter(lambda c: c and lf in c[-1].icons, p.op.board), ps=False, \
        text=comment(TRANSFER, may=False, icon=lf, from_=BOARD, to=his(BOARD)))
    if col:
        p.btbTransfer(col)
    col = p.op.chsST(filter(lambda c: c and lf not in c[-1].icons, p.board), ps=False, op=True, \
        text=comment(TRANSFER, may=False, custom="without a leaf", from_=None, to="your board"))
    if col:
        p.op.btbTransfer(col)

def composites(p):
    c = p.op.chsC(ps=False)
    for cd in p.op.cards:
        if cd != c:
            p.nbTransfer(p.op.cards, p.cards, cd)
    c = p.op.chsSC(filter(lambda c: c.age == p.op.sages[-1], p.op.scores), ps=False, \
        text=comment(TRANSFER, may=False, num=HIGHEST, from_=SCOREPILE, to=his(SCOREPILE)))
    if c:
        p.nbTransfer(p.op.scores, p.scores, c)

def computers(p):
    p.maySplay((1, 2), 3, m=True)

def computers2(p):
    c = p.drawAndMeld(10, rtrn=True)
    p.xSharedDogma(c.color)

def construction(p):
    cs = p.op.chsAC(mx=2, ps=False, text=comment(TRANSFER, may=False, num=2, to=his(HAND)))
    for c in cs:
        p.nbTransfer(p.op.cards, p.cards, c)
    p.op.draw(2)

def construction2(p):
    if not all(p.op.board) and all(p.board):
        p.specAchieve(1)

def corporations(p):
    col = p.op.chsST(filter(lambda c: c and fc in c[-1].icons and c[-1].color != 2, p.op.board), \
        ps=False, text=comment(TRANSFER, may=False, color=non(GREEN), icon=fc, from_=BOARD, to=his(SCOREPILE)))
    if col is not None:
        p.btnbTransfer(p.scores, col)
        p.op.drawAndMeld(8)

def corporations2(p):
    p.drawAndMeld(8)

def currency(p):
    cs = p.chsAC(text=comment(RETURN, num=ANY))
    s = set()
    for c in cs:
        p.rtrn(c)
        s.add(c.age)
    for i in range(len(s)):
        p.drawAndScore(2)

def databases(p):
    for i in range((len(p.op.scores) + 1) // 2):
        c = p.op.chsS(ps=False, text=comment(RETURN, may=False, from_=SCOREPILE))
        p.op.scoreRtrn(c)

def democracy(p):
    cs = p.chsAC(text=comment(RETURN, num=ANY))
    for c in cs:
        p.rtrn(c)
    if cs and ('Democracy' not in p.ps.specs or len(cs) > p.ps.specs['Democracy']):
        p.drawAndScore(8)
        p.ps.specs['Democracy'] = len(cs)

def domestication(p):
    c = p.chsSC(filter(lambda c: c.age == p.cages[0], p.cards), ps=False, \
        text=comment(MELD, may=False, age=LOWEST))
    if c:
        p.meld(c)
    p.draw(1)

def ecology(p):
    c = p.chsC(text=comment(RETURN))
    if c:
        p.rtrn(c)
        c = p.chsC(ps=False, text=comment(SCORE, may=False))
        if c:
            p.score(c)
        p.draw(10)
        p.draw(10)

def education(p):
    c = p.chsSC(filter(lambda c: c.age == p.sages[-1], p.scores), \
        text=comment(RETURN, age=HIGHEST, from_=SCOREPILE))
    if c:
        p.scoreRtrn(c)
        if p.sages:
            p.draw(min(p.sages[-1] + 2, 11))

def electricity(p):
    cs = list(filter(lambda c: c and fc not in c[-1].icons, p.board))
    for i in range(len(cs)):
        c = p.chsST(cs, ps=False, text=comment(RETURN, may=False, noIcon=fc, from_=BOARD))
        cs.remove(p.board[c])
        p.boardRtrn(c)
        p.draw(8)

def emancipation(p):
    c = p.op.chsC(ps=False, text=comment(TRANSFER, may=False, to=his(SCOREPILE)))
    if c:
        p.nbTransfer(p.op.cards, p.scores, c)
        p.op.draw(6)

def emancipation2(p):
    p.maySplay((1, 4), 2, m=True)

def empiricism(p):
    cols = p.chsAT(mx=2, ps=False, text="You must choose 2 colors")
    c = p.drawAndReveal(9)
    if c.color in cols:
        p.meld(c)
        if len(p.board[c.color]) >= 2 and not p.isSplay(c.color, 3):
            p.splay(c.color, 3)

def empiricism2(p):
    if p.icons[lb] >= 20:
        p.win('Empiricism')

def encyclopedia(p):
    if p.scores and p.chsYn(text=comment(MELD, num=HIGHEST, from_=SCOREPILE)):
        cs = list(filter(lambda c: c.age == p.sages[-1], p.scores))
        for i in range(len(cs)):
            c = p.chsSC(cs, ps=False, text=comment(MELD, may=False, num=HIGHEST, from_=SCOREPILE))
            p.meld(c, score=True)
            cs.remove(c)

def engineering(p):
    cols = filter(lambda c: p.op.board[c] and ct in p.op.board[c][-1].icons, range(5))
    for c in cols:
        p.btnbTransfer(p.scores, c)

def engineering2(p):
    p.maySplay(1, 1)

def enterprise(p):
    col = p.op.chsST(filter(lambda c: c and cr in c[-1].icons, p.op.board[:4]), ps=False, \
        text=comment(TRANSFER, may=False, color=non(PURPLE), icon=cr, from_=BOARD, to=his(BOARD)))
    if col:
        p.btbTransfer(col)
        p.op.drawAndMeld(4)

def enterprise2(p):
    p.maySplay(2, 2)

def evolution(p):
    if p.chsYn(text="You may choose to execute the dogma effect"):
        if p.chsYn(text=comment(SCORE, drawAnd=True, age=8, custom="and then return a card from your score pile")):
            p.drawAndScore(8)
            c = p.chsS(ps=False, text=comment(RETURN, may=False, from_=SCOREPILE))
            p.scoreRtrn(c)
        elif p.scores:
            p.draw(p.sages[-1])
        else:
            p.draw(1)

def experimentation(p):
    p.drawAndMeld(5)

def explosives(p):
    if p.op.cards:
        #print(p.op.cages, p.op.cages[-3:])
        for a in dc(p.op.cages[-3:]):
            c = p.op.chsSC(filter(lambda c: c.age == a, p.op.cards), ps=False, \
                text=comment(TRANSFER, may=False, age=a, to=HAND))
            #print(p.op.cards, c, a)
            p.nbTransfer(p.op.cards, p.cards, c)
        if not p.op.cards:
            p.op.draw(7)

def fermenting(p):
    for i in range(p.icons[lf] // 2):
        p.draw(2)

def feudalism(p):
    c = p.op.chsSC(filter(lambda c: ct in c.icons, p.op.cards), ps=False, \
        text=comment(TRANSFER, may=False, icon=ct, to=his(HAND)))
    if c:
        p.nbTransfer(p.op.cards, p.cards, c)

def feudalism2(p):
    p.maySplay((3, 4), 1, m=True)

def fission(p):
    c = p.op.drawAndReveal(10)
    if c.color == 1:
        for sp in [p, p.op]:
            sp.cards = []
            sp.scores = []
            sp.cages = []
            sp.sages = []
            for c in sp.board:
                c.clear()
        p.ps.dones.add('Fission')

def fission2(p):
    if 'Fission' not in p.ps.dones:
        if p.chsYn(text="You must choose whether to return a top card from his board"):
            col = p.chsST(filter(lambda c: c, p.op.board), ps=False, op=True, \
                text=comment(RETURN, may=False, custom="from his board", from_=None))
            if col:
                p.op.boardRtrn(col)
        else:
            col = p.chsST(filter(lambda c: c and c[-1].name != 'Fission', p.board), ps=False, \
                text=comment(RETURN, may=False, color=non(RED), from_=BOARD))
            if col:
                p.boardRtrn(col)

def flight(p):
    if p.isSplay(1, 3):
        p.maySplay((0, 2, 3, 4), 3, m=True)

def flight2(p):
    p.maySplay(1, 3)

def genetics(p):
    c = p.drawAndMeld(10, rtrn=True)
    for cd in [p.board[c.color][i] for i in range(len(p.board[c.color])-1)]:
        c = p.op.chsSC(p.board[c.color], ps=False)
        p.boardScore(c.color, card=c)

def globalization(p):
    col = p.op.chsST(filter(lambda c: c and lf in c[-1].icons, p.op.board), ps=False)
    if col is not None:
        p.op.boardRtrn(col)

def globalization2(p):
    p.drawAndScore(6)
    if p.icons[fc] >= p.icons[lf] and p.op.icons[fc] >= p.op.icons[fc]:
        s = sum(p.sages)
        os = sum(p.op.sages)
        if s > os:
            p.win('Globalization')
        elif os > s:
            p.op.win('Globalization')

def gunpowder(p):
    col = p.op.chsST(filter(lambda c: c and ct in c[-1].icons, p.op.board), ps=False)
    if col is not None:
        p.btnbTransfer(p.scores, col)
        p.ps.dones.add('Gunpowder')

def gunpowder2(p):
    if 'Gunpowder' in p.ps.dones:
        p.drawAndScore(2)

def industrialization(p):
    for i in range(p.icons[fc] // 2):
        p.drawAndTuck(6)

def industrialization2(p):
    p.maySplay((1, 4), 2, m=True)

def invention(p):
    col = p.chsST(filter(lambda c: c and p.isSplay(p.board.index(c), 1), p.board))
    if col:
        p.splay(col, 2)
        p.drawAndScore(4)

def invention2(p):
    if not list(filter(lambda c: p.isSplay(c, 0), range(5))):
        p.specAchieve(3)

def lighting(p):
    cs = p.chsAC(mx=3)
    for c in cs:
        p.rtrn(c)
    for _ in set(map(lambda c: c.age, cs)):
        p.drawAndScore(7)

def machinery(p):
    cs = filter(lambda c: c.age == p.cages[-1], p.cards)
    p.exchange(p.cards, p.op.cards, cs, p.op.cards)

def machinery2(p):
    c = p.chsSC(filter(lambda c: ct in c.icons, p.cards), ps=False)
    if c:
        p.score(c)
    else:
        p.revealAll(p.cards)
    p.maySplay(1, 1)

def machineTools(p):
    if p.scores:
        p.drawAndScore(p.sages[-1])
    else:
        p.drawAndScore(1)

def mapmaking(p):
    c = p.op.chsSC(filter(lambda s: s.age == 1, p.op.scores), ps=False)
    if c:
        p.nbTransfer(p.op.scores, p.scores, c)
        p.ps.dones.add('Mapmaking')

def mapmaking2(p):
    if 'Mapmaking' in p.ps.dones:
        p.drawAndScore(1)

def masonry(p):
    cs = p.chsASC(filter(lambda c: ct in c.icons, p.cards))
    for c in cs:
        p.meld(c)
    if len(cs) >= 4:
        p.specAchieve(0)

def massMedia(p):
    c = p.chsC()
    if c:
        p.rtrn(c)
        a = p.chsA()
        cs = list(filter(lambda c: c.age == a, p.op.scores + p.scores))
        for i in range(len(cs)):
            c = p.chsSC(cs, ps=False)
            if c in p.scores:
                p.scoreRtrn(c)
            else:
                p.op.scoreRtrn(c)
            cs.remove(c)

def massMedia2(p):
    p.maySplay(4, 3)

def mathematics(p):
    c = p.chsC()
    if c:
        p.rtrn(c)
        p.drawAndMeld(c.age + 1)

def measurement(p):
    c = p.chsC()
    if c:
        p.rtrn(c)
        col = p.chsT()
        if col is not None:
            p.maySplay(col, 2)
            p.draw(min(len(p.board[col]), 1))

def medicine(p):
    c = p.chsSC(filter(lambda c: c.age == p.sages[0], p.scores), ps=False) if p.scores else None
    oc = p.op.chsSC(filter(lambda c: c.age == p.op.sages[-1], p.op.scores), ps=False) if p.op.scores else None
    if c and oc:
        p.exchange1(c, oc, p.scores, p.op.scores)
    elif c:
        p.op.nbTransfer(p.scores, p.op.scores, c)
    elif oc:
        p.nbTransfer(p.op.scores, p.scores, oc)

def metalworking(p):
    c = p.drawAndReveal(1)
    while ct in c.icons:
        p.score(c)
        c = p.drawAndReveal(1)

def metricSystem(p):
    if p.isSplay(2, 2):
        p.maySplay((0, 1, 3, 4), 2, m=True)

def metricSystem2(p):
    p.maySplay(2, 2)

def miniaturization(p):
    c = p.chsC()
    if c:
        p.rtrn(c)
        if c.age == 10:
            for i in set(p.sages):
                p.draw(10)

def mobility(p):
    cs = list(filter(lambda c: c and fc not in c[-1].icons and c[-1].color != 1, p.op.board))
    ages = list(map(lambda c: c[-1].age, cs))
    ages.sort()
    for a in ages[-2:]:
        col = p.op.chsST(filter(lambda c: c[-1].age == a, cs), ps=False)
        cs.remove(p.op.board[col])
        p.btbTransfer(col)
    if ages:
        p.op.draw(8)

def monotheism(p):
    col = p.op.chsST(filter(lambda c: not p.board[p.op.board.index(c)] and c, p.op.board), ps=False)
    if col:
        p.btnbTransfer(p.scores, col)
        p.op.drawAndTuck(1)

def monotheism2(p):
    p.drawAndTuck(1)

def mysticism(p):
    c = p.drawAndReveal(1)
    if p.board[c.color]:
        p.meld(c)
        p.draw(1)

def navigation(p):
    c = p.op.chsSC(filter(lambda c: c.age == 2 or c.age == 3, p.op.scores), ps=False)
    if c:
        p.nbTransfer(p.op.scores, p.scores, c)

def oars(p):
    op = p.op
    c = op.chsSC(filter(lambda c: cr in c.icons, op.cards), ps=False)
    if c:
        p.nbTransfer(op.cards, p.scores, c)
        op.draw(1)
        p.ps.dones.add('Oars')

def oars2(p):
    if 'Oars' not in p.ps.dones:
        p.draw(1)

def optics(p):
    c = p.drawAndMeld(3, rtrn=True)
    if cr in c.icons:
        p.drawAndScore(4)
    elif sum(p.sages) > sum(p.op.sages):
        c = p.chsS(ps=False)
        p.op.nbTransfer(p.scores, p.op.scores, c)

def paper(p):
    p.maySplay((0, 2), 1, m=True)

def paper2(p):
    cols = filter(lambda c: p.isSplay(c, 1), range(5))
    for i in cols:
        p.draw(4)

def perspective(p):
    c = p.chsC()
    if c:
        p.rtrn(c)
        for i in range(p.icons[lb] // 2):
            c = p.chsC(ps=False)
            if c:
                p.score(c)
            else:
                break

def philosophy(p):
    p.maySplay(list(range(5)), 1, m=True)

def philosophy2(p):
    c = p.chsC()
    if c:
        p.score(c)

def physics(p):
    cs = set([p.drawAndReveal(6).color, p.drawAndReveal(6).color, p.drawAndReveal(6).color])
    if len(cs) < 3:
        for i in range(len(p.cards)):
            c = p.chsC(ps=False)
            p.rtrn(c)

def pottery(p):
    cs = p.chsAC(mx=3)
    if cs:
        for c in cs:
            p.rtrn(c)
        p.drawAndScore(len(cs))

def pottery2(p):
    p.draw(1)

def printingPress(p):
    c = p.chsS()
    if c:
        p.scoreRtrn(c)
        p.draw(min((p.board[4][-1].age if p.board[4] else 0) + 2, 11))

def printingPress2(p):
    p.maySplay(0, 2)

def publications(p):
    col = p.chsST(filter(lambda c: len(c) > 1, p.board))
    if col is not None:
        for i in range(len(p.board[col]), 0, -1):
            c = p.chsSC([p.board[col][j] for j in range(i)], ps=False)
            p.meld(c, frm=p.board[col])
        p.ckSpec([1, 2, 4])

def publications2(p):
    p.maySplay((0, 3), 3, m=True)

def quantumTheory(p):
    cs = p.chsAC(mx=2)
    for c in cs:
        p.rtrn(c)
    if len(cs) == 2:
        p.draw(10)
        p.drawAndScore(10)

def railroad(p):
    for i in range(len(p.cards)):
        c = p.chsC(ps=False)
        p.rtrn(c)
    p.draw(6)
    p.draw(6)
    p.draw(6)

def railroad2(p):
    cols = tuple(filter(lambda col: p.isSplay(col, 2), range(5)))
    p.maySplay(cols, 3, m=True)

def reformation(p):
    if p.chsYn():
        for i in range(p.icons[lf] // 2):
            c = p.chsC(ps=False)
            if c:
                p.tuck(c)
            else:
                break

def reformation2(p):
    p.maySplay((3, 4), 2, m=True)

def refrigeration(p):
    cs = p.op.chsAC(mx=len(p.op.cards) // 2, ps=False)
    for c in cs:
        p.op.rtrn(c)

def refrigeration2(p):
    c = p.chsC()
    if c:
        p.score(c)

def roadBuilding(p):
    cs = p.chsAC(mx=2)
    for c in cs:
        p.meld(c)
    if len(cs) == 2 and p.chsYn():
        if p.board[1]:
            p.op.btbTransfer(1)
        if p.op.board[2]:
            p.btbTransfer(2)

def robotics(p):
    if p.board[2]:
        p.boardScore(2)
    c = p.drawAndMeld(10, rtrn=True)
    p.xSharedDogma(c.color)

def rocketry(p):
    for i in range(p.icons[cl] // 2):
        c = p.chsSC(p.op.scores, ps=False)
        if c:
            p.op.scoreRtrn(c)
        else:
            break

def sailing(p):
    p.drawAndMeld(1)

def sanitation(p):
    cs = []
    #print(p.op.cards, p.op.cages)
    for a in dc(p.op.cages[-2:]):
        c = p.op.chsSC(filter(lambda c: c.age == a and c not in cs, p.op.cards), ps=False)
        cs.append(c)
    c = p.op.chsSC(filter(lambda c: c.age == p.cages[0], p.cards), ps=False)
    if c:
        c = [c]
    else:
        c = []
    p.exchange(p.op.cards, p.cards, cs, c)

def satellites(p):
    for i in range(len(p.cards)):
        c = p.chsC(ps=False)
        p.rtrn(c)
    p.draw(8)
    p.draw(8)
    p.draw(8)

def satellites2(p):
    p.maySplay(4, 3)

def satellites3(p):
    c = p.chsC(ps=False)
    p.meld(c)
    p.xSharedDogma(c.color)

def selfService(p):
    col = p.chsST(filter(lambda c: c and c[-1].name != 'Selfservice', p.board), ps=False)
    if col is not None:
        p.xSharedDogma(col)

def selfService2(p):
    if len(p.achs) > len(p.op.achs):
        p.win('Selfservice')

def services(p):
    cs = list(filter(lambda c: c.age == p.op.sages[-1], p.op.scores))
    for c in cs:
        p.nbTransfer(p.op.scores, p.cards, c)
    if cs:
        col = p.op.chsST(filter(lambda c: c and lf not in c[-1].icons, p.board), ps=False, op=True)
        if col is not None:
            p.op.btbTransfer(col)

def skyscrapers(p):
    col = p.op.chsST(filter(lambda c: c and cl in c[-1].icons and c[-1].color != 3, p.op.board), ps=False)
    if col is not None:
        p.btbTransfer(col)
        if p.op.board[col]:
            p.op.boardScore(col)
            for i in range(len(p.op.board[col])):
                c = p.op.chsSC(p.op.board[col], ps=False)
                p.op.boardRtrn(col, card=c)

def socialism(p):
    if p.cards and p.chsYn():
        pp = False
        for i in range(len(p.cards)):
            c = p.chsC(ps=False)
            p.tuck(c)
            if c.color == 4:
                pp = True
        if pp:
            cs = list(filter(lambda c: c.age == p.cages[0], p.cards))
            for c in cs:
                p.nbTransfer(p.op.cards, p.cards, c)

def societies(p):
    col = p.op.chsST(filter(lambda c: c and lb in c[-1].icons and c[-1].color < 4, p.op.board), ps=False)
    if col is not None:
        p.btbTransfer(col)
        p.op.draw(5)

def software(p):
    p.drawAndScore(10)

def software2(p):
    p.drawAndMeld(10)
    c = p.drawAndMeld(10, rtrn=True)
    p.xSharedDogma(c.color)

def specialization(p):
    c = p.chsC(ps=False)
    if c:
        p.reveal(c)
        if p.op.board[c.color]:
            p.btnbTransfer(p.cards, c.color)

def specialization2(p):
    p.maySplay((0, 3), 3, m=True)

def statistics(p):
    c = p.op.chsSC(filter(lambda c: c.age == p.op.sages[-1], p.op.scores), ps=False)
    if c:
        p.op.nbTransfer(p.op.scores, p.op.cards, c, s=True)
        if len(p.op.cards) == 1:
            c = p.op.chsSC(filter(lambda c: c.age == p.op.sages[-1], p.op.scores), ps=False)
            if c:
                p.op.nbTransfer(p.op.scores, p.op.cards, c, s=True)

def statistics2(p):
    p.maySplay(3, 2)

def steamEngine(p):
    cs = [p.drawAndReveal(4), p.drawAndReveal(4)]
    c = p.chsSC(cs, ps=False)
    p.tuck(c)
    cs.remove(c)
    p.tuck(cs[0])
    if p.board[3]:
        p.boardScore(3, idx=-1)

def stemCells(p):
    if p.chsYn():
        for c in p.cards:
            p.score(c)

def suburbia(p):
    cs = p.chsAC()
    for c in cs:
        p.tuck(c)
    for i in range(len(cs)):
        p.drawAndScore(1)

def theInternet(p):
    p.maySplay(2, 3)

def theInternet2(p):
    p.drawAndScore(10)

def theInternet3(p):
    for i in range(p.icons[cl] // 2):
        p.drawAndMeld(10)

def thePirateCode(p):
    cs = p.op.chsASC(filter(lambda c: c.age < 5, p.op.scores), mx=2, ps=False)
    for c in cs:
        p.nbTransfer(p.op.scores, p.scores, c)
        p.ps.dones.add('Thepiratecode')

def thePirateCode2(p):
    if 'Thepiratecode' in p.ps.dones:
        crs = list(filter(lambda c: c and cr in c[-1].icons, p.board))
        if crs:
            hage = max(map(lambda c: c[-1].age, crs))
            col = p.chsST(filter(lambda c: c[-1].age == hage, crs), ps=False)
            p.boardScore(col)

def theWheel(p):
    p.draw(1)
    p.draw(1)

def tools(p):
    cs = p.chsAC(mx=3)
    for c in cs:
        p.rtrn(c)
    if len(cs) == 3:
        p.drawAndMeld(3)

def tools2(p):
    c = p.chsSC(filter(lambda c: c.age == 3, p.cards))
    if c:
        p.rtrn(c)
        p.draw(1)
        p.draw(1)
        p.draw(1)

def translation(p):
    if p.scores and p.chsYn():
        for c in p.scores:
            p.meld(c, score=True)

def translation2(p):
    if not list(filter(lambda c: c and cr not in c[-1].icons, p.board)):
        p.specAchieve(2)

def vaccination(p):
    if p.op.scores:
        cs = list(filter(lambda c: c.age == p.op.sages[0], p.op.scores))
        for i in range(len(cs)):
            c = p.op.chsSC(cs, ps=False)
            p.op.scoreRtrn(c)
            cs.remove(c)
        p.op.drawAndMeld(6)
        p.ps.dones.add('Vaccination')

def vaccination2(p):
    if 'Vaccination' in p.ps.dones:
        p.drawAndMeld(7)

def writing(p):
    p.draw(2)

def monument(p):
    return p.scored >= 6 or p.tucked >= 6

def empire(p):
    return min(p.icons.values()) >= 3

def world(p):
    return p.icons[cl] >= 12

def wonder(p):
    return min(p.splays) >= 2

def universe(p):
    return all(map(lambda t: t and t[-1].age >= 8, p.board))

ages = _it.chain.from_iterable([_it.repeat(1, 5)] + [_it.repeat(i, 10) for i in range(1, 11)])
colors = _it.chain(_it.chain.from_iterable([_it.repeat(i, 3) for i in range(5)]), _it.cycle(_it.chain.from_iterable([_it.repeat(i, 2) for i in range(5)])))
ics = [(e,lf,lf,lf), (e,lb,lb,ct), (e,lb,lb,cr), (ct,lb,e,ct), (ct,ct,e,ct), (ct,cr,e,ct), (e,cr,lf,lf), (cr,cr,e,lf), (e,ct,ct,ct), (e,lf,lf,lf), (ct,cr,e,ct), (ct,e,ct,ct), (e,cr,cr,ct), (e,cr,cr,lf), (e,ct,ct,ct),
         (e,lf,lf,lb), (e,lb,cr,lb), (ct,e,ct,ct), (ct,ct,e,ct), (lf,cr,e,cr), (e,cr,cr,ct), (e,cr,lf,cr), (lf,lf,e,ct), (e,ct,ct,ct), (e,lb,lb,lb),
         (e,lf,ct,ct), (e,cr,cr,cr), (ct,e,lb,ct), (cr,cr,cr,e), (e,cr,cr,lf), (e,lb,lb,cr), (lf,lf,e,ct), (cr,lf,lf,e), (lb,lb,lb,e), (e,ct,lf,ct),
         (e,lb,lb,lb), (e,lb,lb,cr), (e,fc,lb,fc), (e,fc,cr,fc), (e,lb,lb,fc), (e,cr,cr,cr), (lf,lf,lf,e), (e,lb,lb,lf), (e,cr,cr,cr), (lf,lf,e,lf),
         (fc,lb,fc,e), (fc,lb,lb,e), (fc,fc,fc,e), (cr,fc,cr,e), (fc,cr,e,cr), (lb,lf,lb,e), (lf,lb,lf,e), (e,fc,cr,fc), (cr,lb,lb,e), (cr,e,lb,cr),
         (lb,lb,lb,e), (e,cr,cr,cr), (cr,fc,fc,e), (fc,fc,e,fc), (lb,lb,lb,e), (e,fc,cr,cr), (e,fc,lf,fc), (lf,fc,lf,e), (cr,lb,lb,e), (fc,lb,fc,e),
         (lb,lb,lb,e), (e,lb,cl,lb), (cr,cr,fc,e), (e,fc,fc,fc), (cr,cr,cl,e), (lb,fc,e,fc), (e,lf,lf,cr), (lf,lf,e,lf), (e,lf,cl,lf), (cl,fc,cl,e),
         (cl,cl,cl,e), (cl,cl,cl,e), (cr,e,cl,cr), (e,fc,cl,fc), (e,fc,fc,cr), (lb,e,cl,lb), (lf,lf,lf,e), (fc,cr,cr,e), (lb,lb,lb,e), (lf,e,lf,lf),
         (cl,e,cl,fc), (lb,lb,lb,e), (fc,fc,e,fc), (e,cl,cl,cl), (e,cr,cl,cr), (e,cl,cl,cl), (lf,lb,lb,e), (e,cr,lf,lf), (e,lf,lf,lf), (e,fc,lf,fc),
         (lb,cl,cl,e), (cl,cl,cl,e), (e,lb,cl,lb), (e,fc,cl,fc), (e,cl,cl,cl), (e,cr,cr,cr), (e,fc,fc,fc), (e,lf,lf,lf), (lb,lb,cl,e), (e,cl,cl,lb)]
funcs = [[pottery, pottery2], [tools, tools2], [writing], [archery], [metalworking], [oars, oars2], [clothing, clothing2], [sailing], [theWheel], [agriculture], [domestication], [masonry], [cityStates], [codeOfLaws], [mysticism],
         [calendar], [mathematics], [construction, construction2], [roadBuilding], [currency], [mapmaking, mapmaking2], [canelBuilding], [fermenting], [monotheism, monotheism2], [philosophy, philosophy2],
         [alchemy, alchemy2], [translation, translation2], [engineering, engineering2], [optics], [compass], [paper, paper2], [machinery, machinery2], [medicine], [education], [feudalism, feudalism2],
         [experimentation], [printingPress, printingPress2], [colonialism], [gunpowder, gunpowder2], [invention, invention2], [navigation], [anatomy], [perspective], [enterprise, enterprise2], [reformation, reformation2],
         [chemistry, chemistry2], [physics], [coal, coal2, coal3], [thePirateCode, thePirateCode2], [banking, banking2], [measurement], [statistics, statistics2], [steamEngine], [astronomy, astronomy2], [societies],
         [atomicTheory, atomicTheory2], [encyclopedia], [industrialization, industrialization2], [machineTools], [classfication], [metricSystem, metricSystem2], [canning, canning2], [vaccination, vaccination2], [democracy], [emancipation, emancipation],
         [evolution], [publications, publications2], [combustion], [explosives], [bicycle], [electricity], [refrigeration, refrigeration2], [sanitation], [lighting], [railroad, railroad2],
         [quantumTheory], [rocketry], [flight, flight2], [mobility], [corporations, corporations2], [massMedia, massMedia2], [antibiotics], [skyscrapers], [empiricism, empiricism2], [socialism],
         [computers, computers2], [genetics], [composites], [fission, fission2], [collaboration, collaboration2], [satellites, satellites2, satellites3], [ecology], [suburbia], [services], [specialization, specialization2],
         [bioengineering, bioengineering2], [software, software2], [miniaturization], [robotics], [databases], [selfService, selfService2], [globalization, globalization2], [stemCells], [ai, ai2], [theInternet, theInternet2, theInternet3]]
dems = [[f, f], [f, f], [f], [t], [f], [t, f], [f, f], [f], [f], [f], [f], [f], [t], [f], [f],
        [f], [f], [t, f], [f], [f], [t, f], [f], [f], [t, f], [f, f],
        [f, f], [f, f], [t, f], [f], [t], [f, f], [t, f], [t], [f], [t, f],
        [f], [f, f], [f], [t, f], [f, f], [t], [t], [f], [t, f], [f, f],
        [f, f], [f], [f, f, f], [t, f], [t, f], [f], [t, f], [f], [f, f], [t],
        [f, f], [f], [f, f], [f], [f], [f, f], [f, f], [t, f], [f], [t, f],
        [f], [f, f], [t], [t], [f], [f], [t, f], [t], [f], [f, f],
        [f], [f], [f, f], [t], [t, f], [f, f], [f], [t], [f, f], [f],
        [f, f], [f], [t], [t, f], [t, f], [f, f, f], [f], [f], [t], [f, f],
        [f, f], [f, f], [f], [f], [t], [f, f], [t, f], [f], [t], [f, f, f]]
msgs = []
idxs = _it.count()
c = 105
spec_names = ['monument', 'empire', 'world', 'wonder', 'universe']
conds = [monument, empire, world, wonder, universe]
