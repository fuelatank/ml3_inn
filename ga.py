import random

from rule import Players, Game
from agents import ModelAgent, RandomAgent
from models import PolicyModel
from env import Computer
import trainconfig as tc

model1 = PolicyModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
model2 = PolicyModel(296, 220, tc.RNN_LAYERS, tc.LEARNING_RATE)
agent1 = ModelAgent(model1)
agent2 = ModelAgent(model2)
agents = [agent1, agent2]
player1 = Computer(agent1, name='ga1')
player2 = Computer(agent2, name='ga2')
evalgame = Game(path='log\\qAgents\\test\\Game\\0\\')
evaluators = [Computer(RandomAgent(), name='rand')]

sample_weights = model1.get_weights()

population = [sample_weights.random_copy() for _ in range(tc.POPULATION_SIZE)]

def train(n=tc.GA_EPISODES):
    for _ in range(n):
        i1, i2 = random.sample(range(tc.POPULATION_SIZE), k=2)
        w1 = population[i1]
        w2 = population[i2]
        model1.set_weights(w1)
        model2.set_weights(w2)
        r = evalgame.run(player1, player2)
        if r == player1:
            population[i2] = w1.noise_copy()
        elif r == player2:
            population[i1] = w2.noise_copy()
        else:
            population[i1] = w1.noise_copy()