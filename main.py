
from rule import Players, Game
from agents import QAgent, QBuffer, RandomAgent
from models import QModel
from env import Computer

EVAL_N_PER_AGENT = 20

model1 = QModel(296, 220, [64], 3e-5)
model2 = QModel(296, 220, [64], 3e-5)
buffer1 = QBuffer(20)
buffer2 = QBuffer(20)
agent1 = QAgent(model1, buffer1)
agent2 = QAgent(model2, buffer2)
agents = [agent1, agent2]
player1 = Computer(agent1, name='q1')
player2 = Computer(agent2, name='q2')
players = Players(player1, player2, path='log\\qAgents\\test\\0\\')
#rand = Computer(RandomAgent(), name='rand')
#rand2 = Computer(RandomAgent(), name='rand')
#players = Players(rand, rand2, path='log\\qAgents\\test\\0\\')
#evalgame = Game(path='log\\qAgents\\test\\Game\\0\\')
#evaluators = [Computer(RandomAgent(), name='rand')]
for i in range(1000):
	print(f'collecting trajectory {i+1}')
	r = players.run()
	players.reset()
	if r == player1:
		v = 1
	elif r == player2:
		v = -1
	else:
		v = 0
	agent1.finish_path(v)
	agent2.finish_path(-v)
	agent1.train()
	agent2.train()
	if i % 10 == 0: pass
'''		for e in evaluators:
			win = 0
			lose = 0
			for player in [player1, player2]:
				for j in range(EVAL_N_PER_AGENT):
					if j % 2 == 0:
						r = evalgame.run(e, player)
					else:
						r = evalgame.run(player, e)
					if r == e:
						lose += 1
					elif r == player:
						win += 1
			score = win/(win+lose)
			print(i, score)'''