
import os, time

from rule import Players, Game
from agents import QAgent, QBuffer, RandomAgent
from models import QModel
from env import Computer
from eval import evaluate
import trainconfig as tc

model1 = QModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
model2 = QModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
buffer1 = QBuffer()
buffer2 = QBuffer()
agent1 = QAgent(model1, buffer1)
agent2 = QAgent(model2, buffer2)
agents = [agent1, agent2]
player1 = Computer(agent1, name='q1')
player2 = Computer(agent2, name='q2')
players = [player1, player2]
#players = Players(player1, player2, path='log\\qAgents\\test\\0\\')
#rand = Computer(RandomAgent(), name='rand')
#rand2 = Computer(RandomAgent(), name='rand')
#players = Players(rand, rand2, path='log\\qAgents\\test\\0\\')
evalgame = Game(path='log\\qAgents\\test\\Game\\0\\')
evaluators = [Computer(RandomAgent(), name='rand')]
def train(n=tc.EPISODES):
    scores = []
    for i in range(n):
        print(f'trajectory {i+1}')
        t0 = time.time()
        r = evalgame.run(player1, player2)
        #players.reset()
        if r == player1:
            v = 1
        elif r == player2:
            v = -1
        else:
            v = 0
        agent1.finish_path(v)
        agent2.finish_path(-v)
        print('collect:', time.time()-t0)
        t0 = time.time()
        agent1.train()
        agent2.train()
        print('train:', time.time()-t0)
        if i % tc.EVAL_FREQ == 0:
            score = evaluate(players, agents, evaluators, tc.EVAL_N_PER_AGENT, evalgame)
            scores.append(score)
    return scores

if __name__ == '__main__':
    scores = train()
    print(scores)
