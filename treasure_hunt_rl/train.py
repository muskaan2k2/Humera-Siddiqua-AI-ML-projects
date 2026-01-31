from env import TreasureEnv
from agent import QLearningAgent
from settings import *

def train():
    env = TreasureEnv()
    agent = QLearningAgent()

    for episode in range(EPISODES):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state

    print("Training finished!")
    return agent
