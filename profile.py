
from main import train
from rule import Player, Game
from env import Computer
<<<<<<< HEAD
from models import QModel, BasicModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 5
=======
from models import QModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 10
>>>>>>> e4aeeffa856ef5a0247e60e35f2fd9024e821680

lp = LineProfiler()
#lp.add_function(Game.oneStep)
#lp.add_function(Computer.getObs)
<<<<<<< HEAD
lp.add_function(QModel._step)
lp.add_function(BasicModel.forward)
=======
lp.add_function(QAgent.step)
lp.add_function(QModel.step)
>>>>>>> e4aeeffa856ef5a0247e60e35f2fd9024e821680
lpr = lp(train)
lpr(TESTN)
lp.print_stats()