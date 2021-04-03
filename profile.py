
from main import train
from rule import Players, Game
from models import QModel
from line_profiler import LineProfiler

TESTN = 3

lp = LineProfiler()
lp.add_function(Game.oneStep)
lpr = lp(train)
lpr(TESTN)
lp.print_stats()