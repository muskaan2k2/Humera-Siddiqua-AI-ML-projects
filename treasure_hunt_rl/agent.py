import random
import numpy as np
from settings import *

class QLearningAgent:

    def __init__(self):
        self.q_table = {}
        for i in range(range(GRID_SIZE)):
            for j in range(GRID_SIZE):
                self.q_table[(i, j)] = [0, 0, 0, 0]

    def choose_action(self, state):
        if random.uniform(0, 1) < EPSILON:
            return random.choice(ACTIONS)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        old_q = self.q_table[state][action]
        next_max = max(self.q_table[next_state])
        self.q_table[state][action] = old_q + ALPHA * (
            reward + GAMMA * next_max - old_q
        )
