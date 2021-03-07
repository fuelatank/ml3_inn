#TabNine::semSemantic completion enabled.

from rule import *
from image import getTkImage

import random
from tkinter import *
from line_profiler import LineProfiler as lp

BLACK = '#000000'

ICONSIZE = 60

haveTk  = False

def getTk():
    global tk, opcards, opsa, opscores, opachievements, opboard, opstacks, cdsFrm, \
    board, stacks, popstacks, sa, scores, achievements, cards, choices, hide, haveTk
    tk = Tk()
    tk.geometry('1000x640+400+50')
    opcards = Frame(tk)
    opsa = Frame(tk)
    opscores = Frame(opsa)
    opachievements = Frame(opsa)
    opboard = Frame(tk)
    opstacks = []
    for i in range(5):
        frm = Frame(opboard)
        frm.grid(row=0, column=i+1)
        opstacks.append(frm)
    cdsFrm = Frame(tk)
    cdsFrm.config(height=100)
    board = Frame(tk)
    stacks = []
    for i in range(5):
        frm = Frame(board)
        frm.grid(row=0, column=i+1)
        stacks.append(frm)
    del i
    popstacks = [stacks, opstacks]
    sa = Frame(tk)
    scores = Frame(sa)
    achievements = Frame(sa)
    cards = Frame(tk)
    choices = Frame(tk)
    hide = Frame(tk)
    choices.pack()
    opcards.pack()
    opsa.pack()
    opboard.pack()
    cdsFrm.pack()
    board.pack()
    sa.pack()
    cards.pack()
    opscores.pack(side=RIGHT)
    opachievements.pack(side=LEFT)
    scores.pack(side=LEFT)
    achievements.pack(side=RIGHT)
    Label(opcards, text="Cards: ").grid(column=0)
    Label(opscores, text="Scores: ").grid(column=0)
    Label(opachievements, text="Achievements: ").grid(column=0)
    Label(opboard, text="Board: ").grid(row=0, column=0)
    Label(board, text="Board: ").grid(row=0, column=0)
    Label(scores, text="Scores: ").grid(column=0)
    Label(achievements, text="Achievements: ").grid(column=0)
    Label(cards, text="Cards: ").grid(column=0)
    ctrs = []
    for m in [opcards, opscores]:
        c = []
        for i in range(10):
            c.append(0)
        ctrs.append(c)
    del m

    r = []

    def pss():
        r.append(None)
        tk.quit()

    def yes():
        r.append(True)
        tk.quit()

    def no():
        r.append(False)
        tk.quit()

    topBtns = []
    ageBtns = []
    def btFn(i):
        def cmd():
            r.append(i)
            tk.quit()
        return cmd

    for i in range(16):
        topBtns.append(Button(choices, text=str(i), command=btFn(i)))
    for i in range(1, 11):
        ageBtns.append(Button(choices, text=str(i), command=btFn(i)))

    passBtn = Button(choices, text='pass', command=pss)
    yesBtn = Button(choices, text='yes', command=yes)
    noBtn = Button(choices, text='no', command=no)
    noneBtn = Button(choices, text='change', command=tk.quit)

    def wait():
        noneBtn.pack(fill=X, side=RIGHT)
        tk.mainloop()
        noneBtn.pack_forget()

    def ageLbl(age, master, bd=1):
        return Label(master, text=str(age), bd=bd)

    def boardImg(age, color, icons, name, master):
        img = getTkImage(name, age, icons, color)
        lbl = Label(master, image=img, width=175, height=125)
        lbl.image = img
        return lbl

    specs = list(map(lambda i, name: UserSpec(name, i), range(5), spec_names))
    cds = list(map(lambda c: UserCard2(str(c[3][0])[10:-23], c[0], c[1], c[2], c[3], c[4], c[5]), l))
    acds = {}
    lst = 0
    for a, i in enumerate(range(15, 115, 10), 1):
        acds[a] = cds[lst:i]
        lst = i
    del lst
    haveTk = True

class UserCard2(Card):
    def __init__(self, name, age, color, icons, cmd, dem, idx=None, msg=''):
        super(UserCard2, self).__init__(name, age, color, icons, cmd, dem, msg)
        self.labels = []
        self.idx = idx
        for p in [cards, scores, achievements]:
            f = Label(p, text=self.name, bg=COLORS[self.color], bd=1)
            self.labels.append(f)
        for p in [stacks[color], opstacks[color]]:
            self.labels.append(boardImg(age, color, icons, name, p))
        for p in [opcards, opscores, opachievements]:
            self.labels.append(ageLbl(self.age, p))
        self.crtLabel = None
        self.btn = Button(choices, text=self.name, command=self.chsd)
    
    def chsd(self):
        r.append(self)
        tk.quit()
    
    def cm(self, m, x=None, y=None):
        if m == -1:
            self.hide()
        else:
            l = self.labels[m]
            if self.crtLabel:
                self.crtLabel.grid_forget()
                self.crtLabel.place_forget()
            self.crtLabel = l
            if x is None:
                l.grid(row=0, column=self.idx+1, padx=1)
            else:
                l.place(x=x, y=y, width=175, height=125)
                l.lift()
    
    def hide(self):
        if self.crtLabel:
            self.crtLabel.grid_forget()
            self.crtLabel.place_forget()
        self.crtLabel = None

class UserSpec:
    def __init__(self, name, i, cond=None):
        self.i = i
        self.name = name
        self.cond = cond
        self.labels = {2: Label(achievements, text=self.name, bd=1),
                       7: Label(opachievements, text=self.name, bd=1)}

    def __repr__(self):
        return self.name
    
    def cm(self, m, x=None, y=None):
        l = self.labels[m]
        if self.crtLabel:
            self.crtLabel.grid_forget()
            self.crtLabel.place_forget()
        self.crtLabel = l
        if x is None:
            l.grid(row=0, column=self.idx+1, padx=1)
        else:
            l.place(x=x, y=y, width=175, height=125)
            l.lift()

class Show(Player):
    def __init__(self, name):
        super(Show, self).__init__(name)

    def drawStack(self, col):
        cs = self.board[col]
        self.stacks[col].grid(row=0, column=col+1)
        xb = 0
        yb = 0
        dx = 0
        dy = 0
        spl = self.splays[col]
        if spl == 1:
            xb = (len(cs) - 1) * ICONSIZE
            dx = -ICONSIZE
        elif spl == 2:
            dx = ICONSIZE
        elif spl == 3:
            yb = (len(cs) - 1) * ICONSIZE
            dy = -ICONSIZE
        for i, c in enumerate(cs):
            #c.labels[self.masters[1]].grid(row=yb+dy*i, column=xb+dx*i, rowspan=2, columnspan=3)
            c.cm(self.masters[1], y=yb+dy*i, x=xb+dx*i)
        self.stacks[col].config(width=xb+175, height=yb+125)
        if len(cs) == 0:
            self.stacks[col].grid_forget()

class SimpleRandomComputer(Player):
    def __init__(self, name, probs=defaultdict(lambda: 1)):
        super(SimpleRandomComputer, self).__init__(name)
        self.probs = probs

    def getMove(self):
        l = [0]
        w = [self.probs['draw']]
        for age in self.canAchieve():
            l.append(age)
            w.append(self.probs['achieve'])
        for i, c in enumerate(self.board):
            if c:
                l.append(c[-1].color + 10)
                w.append(self.probs[c[-1].name])
        for c in self.cards:
            l.append(self.mcds.index(c) + 15)
            w.append(self.probs['meld'])
        return random.choices(l, weights=w)[0]

    def chsC(self, ps=True):
        if not self.cards:
            return None
        cards = cp(self.cards)
        if ps:
            cards.append(None)
        return random.choice(cards)
    
    def chsS(self, ps=True):
        if not self.scores:
            return None
        cards = cp(self.scores)
        if ps:
            cards.append(None)
        return random.choice(cards)
    
    def chsSC(self, cs, ps=True):
        l = list(cs)
        if not l:
            return None
        if ps:
            l.append(None)
        return random.choice(l)
    
    def chsAC(self, mx=None, ps=False):
        if not mx:
            mx = len(self.cards)
        cs = []
        chs = cp(self.cards)
        for i in range(mx):
            c = self.chsSC(chs, ps=ps)
            if not c:
                return cs
            cs.append(c)
            chs.remove(c)
        return cs
    
    def chsAS(self, mx=None, ps=False):
        if not mx:
            mx = len(self.scores)
        cs = []
        chs = cp(self.scores)
        for i in range(mx):
            c = self.chsSC(chs, ps=ps)
            if not c:
                return cs
            cs.append(c)
            chs.remove(c)
        return cs
    
    def chsASC(self, cs, mx=None, ps=True):
        l = list(cs)
        if not mx:
            mx = len(l)
        chs = []
        for i in range(mx):
            c = self.chsSC(l, ps=ps)
            if not c:
                return chs
            chs.append(c)
            l.remove(c)
        return chs
    
    def chsT(self, ps=True):
        l = list(range(5))
        if ps:
            l.append(None)
        return random.choice(l)
    
    def chsST(self, cs, ps=True, op=False):
        l = list(cs)
        if op:
            bd = self.op.board
        else:
            bd = self.board
        if not l:
            return None
        r = []
        for c in l:
            r.append(bd.index(c))
        if ps:
            r.append(None)
        return random.choice(r)
    
    def chsAT(self, mx=None, ps=False):
        if not mx:
            mx = 5
        cs = []
        chs = cp(self.board)
        for i in range(mx):
            c = self.chsST(chs, ps=ps)
            if not c:
                return cs
            cs.append(c)
            chs.remove(self.board[c])
        return cs

    def chsA(self):
        return random.randint(1, 10)

    def chsYn(self):
        return random.choice([True, False])

    def _reveal(self, cs):
        pass

class SimpleRandomShow(Show, SimpleRandomComputer):
    def __init__(self, name, probs=lambda: 1):
        super(SimpleRandomShow, self).__init__(name, probs=probs)

class Fast(Player):
    def __init__(self, name, probs=None):
        super(Fast, self).__init__(name)

    def drawStack(self, col):
        pass

class SimpleRandomFast(Fast, SimpleRandomComputer):
    def __init__(self, name, probs=lambda: 1):
        super(SimpleRandomFast, self).__init__(name, probs=probs)

class User(Show):
    def __init__(self, name):
        super(User, self).__init__(name)
        self.drawBtn = topBtns[0]
    
    def getMove(self):
        btns = [self.drawBtn]
        self.drawBtn.pack(fill=X, side=RIGHT)
        for age in self.canAchieve():
            b = topBtns[age]
            b.pack(fill=X, side=RIGHT)
            btns.append(b)
        for i, c in enumerate(self.board):
            if c:
                b = c[-1].btn
                b.pack(fill=X, side=RIGHT)
                btns.append(b)
        for c in self.cards:
            b = c.btn
            b.pack(fill=X, side=RIGHT)
            btns.append(b)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        rs = r.pop()
        if rs in self.cards:
            return cds.index(rs) + 16
        if rs in cds:
            return rs.color + 11
        return rs
    
    def maySplay(self, col_s, d, m=False):
        if not m:
            if len(self.board[col_s]) >= 2 and not self.isSplay(col_s, d) and self.chsYn():
                self.splay(col_s, d)
        else:
            pass
    
    def chsC(self, ps=True):
        if not self.cards:
            return None
        btns = []
        for c in self.cards:
            c.btn.pack(fill=X, side=RIGHT)
            btns.append(c.btn)
        if ps:
            passBtn.pack(fill=X, side=RIGHT)
            btns.append(passBtn)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        return r.pop()
    
    def chsS(self, ps=True):
        if not self.scores:
            return None
        btns = []
        for c in self.scores:
            c.btn.pack(fill=X, side=RIGHT)
            btns.append(c.btn)
        if ps:
            passBtn.pack(fill=X, side=RIGHT)
            btns.append(passBtn)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        return r.pop()
    
    def chsSC(self, cs, ps=True):
        l = list(cs)
        if not l:
            return None
        btns = []
        for c in l:
            c.btn.pack(fill=X, side=RIGHT)
            btns.append(c.btn)
        if ps:
            passBtn.pack(fill=X, side=RIGHT)
            btns.append(passBtn)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        return r.pop()
    
    def chsAC(self, mx=None, ps=False):
        if not mx:
            mx = len(self.cards)
        cs = []
        for i in range(mx):
            c = self.chsC(ps=ps)
            if not c:
                return cs
            cs.append(c)
        return cs
    
    def chsAS(self, mx=None, ps=False):
        if not mx:
            mx = len(self.scores)
        cs = []
        for i in range(mx):
            c = self.chsS(ps=ps)
            if not c:
                return cs
            cs.append(c)
        return cs
    
    def chsASC(self, cs, mx=None, ps=True):
        l = list(cs)
        if not mx:
            mx = len(l)
        chs = []
        for i in range(mx):
            c = self.chsSC(l, ps=ps)
            if not c:
                return chs
            chs.append(c)
            l.remove(c)
        return chs
    
    def chsT(self, ps=True):
        btns = []
        for i in range(5):
            topBtns[i].pack(fill=X, side=RIGHT)
            btns.pack(topBtns[i])
        if ps:
            passBtn.pack(fill=X, side=RIGHT)
            btns.pack(passBtn)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        return r.pop()
    
    def chsST(self, cs, ps=True, op=False):
        l = list(cs)
        if not l:
            return None
        btns = []
        if op:
            bd = self.op.board
        else:
            bd = self.board
        for c in l:
            i = bd.index(c)
            topBtns[i].pack(fill=X, side=RIGHT)
            btns.pack(topBtns[i])
        if ps:
            passBtn.pack(fill=X, side=RIGHT)
            btns.append(passBtn)
        tk.mainloop()
        for b in btns:
            b.pack_forget()
        return r.pop()
    
    def chsA(self):
        for b in ageBtns:
            b.pack(fill=X, side=RIGHT)
        tk.mainloop()
        for b in ageBtns:
            b.pack_forget()
        return r.pop()

    def chsYn(self):
        yesBtn.pack(fill=X, side=RIGHT)
        noBtn.pack(fill=X, side=RIGHT)
        tk.mainloop()
        yesBtn.pack_forget()
        noBtn.pack_forget()
        return r.pop()

class Users(Players):
    def __init__(self, name1, name2):
        super(Users, self).__init__(User(name1), User(name2), cds, 'log\\users\\'+name1+'+'+name2+'\\')
    
    def init(self):
        self.p1.masters = [0, 3, 1, 2]
        self.p2.masters = [5, 4, 6, -1]
        self.p1.stacks = stacks
        self.p2.stacks = opstacks
        self.p1.draw(1, shr=False)
        self.p1.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        c1 = self.p1.chsC(ps=False)
        self.changeTurn()
        c2 = self.p2.chsC(ps=False)
        self.p1.meld(c1, shr=False)
        self.p2.meld(c2, shr=False)
        self.changeTurn()
        del c1, c2
    
    def changeTurn(self):
        self.mp = self.mp.op
        self.p1.masters, self.p2.masters = self.p2.masters, self.p1.masters
        self.p1.stacks, self.p2.stacks = self.p2.stacks, self.p1.stacks
        for p in (self.p1, self.p2):
            for cs in [p.cards, p.scores, p.achs]:
                for c in cs:
                    c.cm(p.masters[p.op.masters.index(c.labels.index(c.crtLabel))])
            for i in range(5):
                p.drawStack(i)
    
    def run(self):
        while not self.oneStep():
            pass

class SimpleRandomComputers(Players):
    def __init__(self, name1, name2, probs1=defaultdict(lambda: 1), probs2=defaultdict(lambda: 1), fast=True):
        self.fast = fast
        if fast:
            super(SimpleRandomComputers, self).__init__(SimpleRandomFast(name1, probs=probs1), SimpleRandomFast(name2, probs=probs2), path='log\\users\\'+name1+'+'+name2+'\\')
        else:
            super(SimpleRandomComputers, self).__init__(SimpleRandomShow(name1, probs=probs1), SimpleRandomShow(name2, probs=probs2), cds=cds, specs=specs, acds=acds, path='log\\users\\'+name1+'+'+name2+'\\')
        self.wins = Counter()
    
    def init(self):
        if not self.fast:
            self.p1.masters = [0, 3, 1, 2]
            self.p2.masters = [5, 4, 6, 7]
            self.p1.stacks = stacks
            self.p2.stacks = opstacks
        self.p1.draw(1, shr=False)
        self.p1.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        self.p2.draw(1, shr=False)
        c1 = self.p1.chsC(ps=False)
        self.changeTurn()
        c2 = self.p2.chsC(ps=False)
        self.p1.meld(c1, shr=False)
        self.p2.meld(c2, shr=False)
        self.changeTurn()
        del c1, c2
    
    def changeTurn(self):
        self.mp = self.mp.op
        if not self.fast:
            self.p1.masters, self.p2.masters = self.p2.masters, self.p1.masters
            self.p1.stacks, self.p2.stacks = self.p2.stacks, self.p1.stacks
            for p in (self.p1, self.p2):
                for cs in [p.cards, p.scores, p.achs]:
                    for c in cs:
                        c.cm(p.masters[p.op.masters.index(c.labels.index(c.crtLabel))])
                for i in range(5):
                    p.drawStack(i)
    
    def run(self):
        while 1:
            r = self.oneStep()
            if r:
                self.wins[r] += 1
                return

d = {'draw': 2, 'meld': 1, 'achieve': 3, 'Agriculture': 0.4599262874902928, 'Codeoflaws': 0.47937555281100397, 'Pottery': 0.5053071785075439, 'Tools': 0.5144951872062961, 'Clothing': 0.5292378717518977, 'Archery': 0.5345543514403449, 'Metalworking': 0.5515089193942482, 'Masonry': 0.6071468236565425, 'Oars': 0.6075559933459621, 'Citystates': 0.6305145562477276, 'Writing': 0.6349609837314356, 'Currency': 0.6826860199587184, 'Perspective': 0.6900313683825071, 'Domestication': 0.6905622043109263, 'Sailing': 0.706883788678707, 'Thewheel': 0.7164869194589648, 'Reformation': 0.7176342338453408, 'Medicine': 0.7210987892086048, 'Colonialism': 0.7463416437168837, 'Mysticism': 0.7532542881813347, 'Canelbuilding': 0.7577218450953958, 'Anatomy': 0.7681398835527554, 'Invention': 0.7741299366646727, 'Gunpowder': 0.7754740731927816, 'Engineering': 0.7815823226713241, 'Feudalism': 0.782267107168167, 'Philosophy': 0.7857095528311347, 'Education': 0.7884859430920251, 'Chemistry': 0.7912825155606286, 'Alchemy': 0.7990790133172381, 'Navigation': 0.8129235283017079, 'Physics': 0.8136658388539412, 'Compass': 0.8166490488358314, 'Banking': 0.8173983635061606, 'Translation': 0.8196548020556259, 'Statistics': 0.8211662144338832, 'Mapmaking': 0.8242062771277162, 'Enterprise': 0.8334668181400334, 'Machinery': 0.8334668181400334, 'Printingpress': 0.849385866326127, 'Mathematics': 0.8542843109100523, 'Monotheism': 0.8542843109100523, 'Optics': 0.8650997337487798, 'Calendar': 0.8659433823199383, 'Metricsystem': 0.8920660282545326, 'Societies': 0.9048215048403047, 'Construction': 0.9334658126161762, 'Paper': 0.9334658126161762, 'Roadbuilding': 0.937427475837556, 'Experimentation': 0.9384233356163063, 'Measurement': 0.9404216399906908, 'Emancipation': 0.9768773081861866, 'Coal': 0.9867309453417149, 'Fermenting': 1.0024762828047005, 'Democracy': 1.0047683804070653, 'Thepiratecode': 1.0059185224244283, 'Astronomy': 1.0271000933811265, 'Canning': 1.0380434336692133, 'Vaccination': 1.0479771104721984, 'Machinetools': 1.0938811485504965, 'Industrialization': 1.1264149924172826, 'Classfication': 1.142704612793874, 'Encyclopedia': 1.1533313845456625, 'Atomictheory': 1.199671771406395, 'Refrigeration': 1.2114535707215244, 'Combustion': 1.2269665300113384, 'Explosives': 1.2269665300113384, 'Railroad': 1.2339971697239225, 'Electricity': 1.2357681990741551, 'Evolution': 1.2357681990741551, 'Bicycle': 1.2465086800571763, 'Lighting': 1.2483180586183453, 'Publications': 1.2723584195878006, 'Sanitation': 1.313349762512007, 'Socialism': 1.4680589147970662, 'Steamengine': 1.481011982813511, 'Skyscrapers': 1.499569560840796, 'Flight': 1.5214086913214286, 'Massmedia': 1.5297784599637962, 'Rocketry': 1.5325907788165676, 'Mobility': 1.5974629055657223, 'Quantumtheory': 1.6130592318993884, 'Antibiotics': 1.758773528603239, 'Corporations': 1.8476214491576743, 'Empiricism': 2.093872779021139, 'Services': 2.1962574777033033, 'Specialization': 2.3784591262997408, 'Fission': 2.4422177165855836, 'Ecology': 2.4504843977286677, 'Composites': 2.4928757084055597, 'Genetics': 2.5191953044709225, 'Suburbia': 2.537129297395932, 'Satellites': 2.564636095435604, 'Collaboration': 2.8761169707976233, 'Computers': 3.126609149108954, 'Databases': 3.2585014630033604, 'Bioengineering': 3.6966057881092853, 'Stemcells': 3.8290280207836807, 'Ai': 3.8861926388754275*2, 'Miniaturization': 3.9460745758980034, 'Selfservice': 5.031596844894339, 'Robotics': 5.103390361352257, 'Software': 5.435565099342625, 'Theinternet': 6.009372768069398, 'Globalization': 6.51721140628397}
d2 = {'Agriculture': 0.21153218992460346, 'Codeoflaws': 0.22980092063285565, 'Pottery': 0.25533534465125485, 'Tools': 0.26470529765844164, 'Clothing': 0.2800927248964782, 'Archery': 0.28574835464380777, 'Metalworking': 0.3041620881714113, 'Masonry': 0.3686272654762287, 'Oars': 0.36912428505059874, 'Citystates': 0.3975486056402688, 'Writing': 0.40317545086119244, 'Currency': 0.4660602018470756, 'Perspective': 0.47614328935183525, 'Domestication': 0.4768761580227655, 'Sailing': 0.49968469069676297, 'Thewheel': 0.5133535057557971, 'Reformation': 0.5149988935867892, 'Medicine': 0.5199834637981159, 'Colonialism': 0.5570258491460198, 'Mysticism': 0.5673920226635691, 'Canelbuilding': 0.5741423945347709, 'Anatomy': 0.5900388807044407, 'Invention': 0.5992771588404501, 'Gunpowder': 0.6013600381942036, 'Engineering': 0.6108709271123018, 'Feudalism': 0.6119418269572525, 'Philosophy': 0.6173395014101017, 'Education': 0.6217100824537203, 'Chemistry': 0.6261280194319564, 'Alchemy': 0.6385272695240507, 'Navigation': 0.6608446628664977, 'Physics': 0.6620520973178877, 'Compass': 0.6669156689644682, 'Banking': 0.6681400846625494, 'Translation': 0.6718339945328473, 'Statistics': 0.6743139517276742, 'Mapmaking': 0.6793159872567297, 'Enterprise': 0.6946669369404715, 'Machinery': 0.6946669369404715, 'Printingpress': 0.7214563499145853, 'Mathematics': 0.7298016838670629, 'Monotheism': 0.7298016838670629, 'Optics': 0.7483975493322097, 'Calendar': 0.7498579413836948, 'Metricsystem': 0.7957817987658166, 'Societies': 0.8187019556214734, 'Construction': 0.8713584233231783, 'Paper': 0.8713584233231783, 'Roadbuilding': 0.8787702724551717, 'Experimentation': 0.8806383568292346, 'Measurement': 0.8843928609627806, 'Emancipation': 0.9542892752490897, 'Coal': 0.9736379584949544, 'Fermenting': 1.0049586975859297, 'Democracy': 1.0095594982658371, 'Thepiratecode': 1.011872073756545, 'Astronomy': 1.0549346018235188, 'Canning': 1.0775341701837704, 'Vaccination': 1.0982560240736583, 'Machinetools': 1.1965759671541534, 'Industrialization': 1.268810735142427, 'Classfication': 1.3057738321003975, 'Encyclopedia': 1.3301732825780148, 'Atomictheory': 1.4392123591093575, 'Refrigeration': 1.4676197540139315, 'Combustion': 1.5054468657680644, 'Explosives': 1.5054468657680644, 'Railroad': 1.5227490148866512, 'Electricity': 1.5271230418429806, 'Evolution': 1.5271230418429806, 'Bicycle': 1.553783889457884, 'Lighting': 1.5582979754726747, 'Publications': 1.6188959478959657, 'Sanitation': 1.7248875986903451, 'Socialism': 2.1551969773151396, 'Steamengine': 2.193396493237208, 'Skyscrapers': 2.2487088678002576, 'Flight': 2.314684406028382, 'Massmedia': 2.340222136569204, 'Rocketry': 2.3488344953135734, 'Mobility': 2.55188773465848, 'Quantumtheory': 2.6019600856158447, 'Antibiotics': 3.093284324915489, 'Corporations': 3.4137050193875047, 'Empiricism': 4.3843032147257075, 'Services': 4.823546908367676, 'Specialization': 5.657067815478526, 'Fission': 5.964427375204502, 'Ecology': 6.0048737835116315, 'Composites': 6.214429297558521, 'Genetics': 6.346344982068344, 'Suburbia': 6.437025071704776, 'Satellites': 6.577358302011181, 'Collaboration': 8.272048829710096, 'Computers': 9.775684771291816, 'Databases': 10.61783178439504, 'Bioengineering': 13.66489435268307, 'Stemcells': 14.66145558394659, 'Ai': 15.102493226449559*2, 'Miniaturization': 15.571504558548607, 'Selfservice': 25.316966809550667, 'Robotics': 26.044593180343117, 'Software': 29.545367949191604, 'Theinternet': 36.11256106561406, 'Globalization': 42.47404451419788, 'draw': 2, 'meld': 1.5, 'achieve': 2}
ps = SimpleRandomComputers('a', 'b', probs1=d, probs2=d2)
def runGame(n):
    for i in range(n):
        ps.run()
        #print(sum(map(lambda c: c.age, ps.p1.scores)), ps.p1.scores)
        #print(sum(map(lambda c: c.age, ps.p2.scores)), ps.p2.scores)
        ps.reset()
lp_ = lp()
#lp_.add_function(SimpleRandomComputers.run)
#lp_.add_function(Players.oneStep)
#lp_.add_function(Players.reset)
#lp_.add_function(SimpleRandomComputer.getMove)
lp_.add_function(Player.makeMove)
'''for c in mainacds[1]:
    for f in c.cmd:
        lp_.add_function(f)'''
#lp_.add_function(Player.draw)
#lp_.add_function(Player.meld)
#lp_.add_function(Player.tuck)
lp_.add_function(Player.cIcon)
lp_.add_function(Player.getIcon)
#lp_.add_function(Player.achieve)
#lp_.add_function(Player.ckSpec)
#lp_.add_function(Player.x)
#lp_.add_function(Player.ailog)
lp_wrapper = lp_(runGame)
lp_wrapper(50)
lp_.print_stats()
