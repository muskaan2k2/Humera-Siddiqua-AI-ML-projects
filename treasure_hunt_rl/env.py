from settings import *

class TreasureEnv:

    def __init__(self):
        self.start = START_STATE
        self.treasure = TREASURE_STATE
        self.trap = TRAP_STATE
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state

        if action == 0:      # UP
            x -= 1
        elif action == 1:    # DOWN
            x += 1
        elif action == 2:    # LEFT
            y -= 1
        elif action == 3:    # RIGHT
            y += 1

        # Boundary check
        if x < 0 or x >= GRID_SIZE or y < 0 or y >= GRID_SIZE:
            return self.state, INVALID_MOVE_REWARD, False

        self.state = (x, y)

        if self.state == self.treasure:
            return self.state, TREASURE_REWARD, True
        elif self.state == self.trap:
            return self.state, TRAP_REWARD, True
        else:
            return self.state, MOVE_REWARD, False
