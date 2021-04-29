import random
import numpy as np

from rule import Players, Game
from agents import ModelAgent, RandomAgent
from models import PolicyModel
from env import Computer
from eval import evaluate
import trainconfig as tc

model1 = PolicyModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
model2 = PolicyModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
agent1 = ModelAgent(model1)
agent2 = ModelAgent(model2)
player1 = Computer(agent1, name='ga1')
player2 = Computer(agent2, name='ga2')
evalgame = Game(path='log\\qAgents\\test\\Game\\0\\')
evaluators = [Computer(RandomAgent(), name='rand')]

sample_weights = model1.get_weights()

population = [sample_weights.random_copy() for _ in range(tc.POPULATION_SIZE)]
winning_streak = [0 for i in range(tc.POPULATION_SIZE)]

def train(n=tc.GA_EPISODES):
    scores = []
    for i in range(n):
        i1, i2 = random.sample(range(tc.POPULATION_SIZE), k=2)
        w1 = population[i1]
        w2 = population[i2]
        model1.set_weights(w1)
        model2.set_weights(w2)
        r = evalgame.run(player1, player2)
        if r == player1:
            population[i2] = w1.noise_copy()
            winning_streak[i2] = winning_streak[i1]
            winning_streak[i1] += 1
        elif r == player2:
            population[i1] = w2.noise_copy()
            winning_streak[i1] = winning_streak[i2]
            winning_streak[i2] += 1
        else:
            population[i1] = w1.noise_copy()
        if i % tc.GA_EVAL_FREQ == 0:
            m = np.argmax(winning_streak)
            w = population[m]
            model1.set_weights(w)
            score = evaluate([player1], [agent1], evaluators, tc.GA_EVAL_PER_AGENT, evalgame)
            scores.append(score)
    return scores

if __name__ == "__main__":
    scores = train()
    print(scores)