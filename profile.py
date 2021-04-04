
from main import train
from rule import Player, Game
from env import Computer
from models import QModel
from agents import QAgent
from line_profiler import LineProfiler

TESTN = 10

lp = LineProfiler()
#lp.add_function(Game.oneStep)
#lp.add_function(Computer.getObs)
lp.add_function(QAgent.step)
lp.add_function(QModel.step)
lpr = lp(train)
lpr(TESTN)
lp.print_stats()