
from main import train
from rule import Player, Game
from env import Computer
from models import QModel, BasicModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 5
from models import QModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 10
from models import QModel, BasicModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 5

lp = LineProfiler()
#lp.add_function(Game.oneStep)
#lp.add_function(Computer.getObs)
lp.add_function(QModel._step)
lp.add_function(BasicModel.forward)
lp.add_function(QAgent.step)
lp.add_function(QModel.step)
lp.add_function(QModel._step)
lp.add_function(BasicModel.forward)
lpr = lp(train)
lpr(TESTN)
lp.print_stats()