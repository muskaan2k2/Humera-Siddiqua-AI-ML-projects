import random

"""
Train an agent to reach the treasure while avoiding the trap by using q learning maximizing the overall reward
"""

GRID = 4

START = (0,0)
TREASURE = (3,3)
TRAP = (1,2)

ACTIONS = ['u','d','l','r']

alpha = 0.1
gama = 0.9
eps = 0.3

EPISODES = 500

def is_terminal(state):
    return state == TREASURE or state == TRAP

def get_reward(state):
    if state == TREASURE:
        return 10
    elif state == TRAP:
        return -10
    else:
        return -1

def move(state, action):
    x, y = state
    if action == 'u':
        x = max(x-1, 0)
    elif action == 'd':
        x = min(x+1, GRID-1)
    elif action == 'l':
        y = max(y-1, 0)
    elif action == 'r':
        y = min(y+1, GRID-1)
    return (x, y)

def initialize_q_table():
    Q = {}
    for i in range(GRID):
        for j in range(GRID):
            Q[(i, j)] = {}
            for a in ACTIONS:
                Q[(i, j)][a] = 0.0
    return Q

def take_action(Q, state):
    if random.uniform(0,1) < eps:
        return random.choice(ACTIONS)
    else:
        return max(Q[state], key=Q[state].get)

def update_q_value(Q, state, action, reward, next_state):
    old_value = Q[state][action]
    future_max = max(Q[next_state].values())
    Q[state][action] = old_value + alpha * (reward + gama * future_max - old_value)

def train_agent():
    Q = initialize_q_table()
    for episode in range(EPISODES):
        state = START
        while not is_terminal(state):
            action = take_action(Q, state)
            next_state = move(state, action)
            reward = get_reward(next_state)
            update_q_value(Q, state, action, reward, next_state)
            state = next_state
    return Q

def test_agent(Q):
    state = START
    path = [state]
    while not is_terminal(state):
        action = max(Q[state], key=Q[state].get)
        state = move(state, action)
        path.append(state)
    return path

if __name__ == "__main__":
    Q = train_agent()
    learned_path = test_agent(Q)
    print("learned_path:")
    print(learned_path)
