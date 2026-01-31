import pygame
import sys
from train import train
from settings import *
from env import TreasureEnv

agent = train()
env = TreasureEnv()

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Treasure Hunt - Q Learning")
clock = pygame.time.Clock()

def draw():
    screen.fill((255, 255, 255))

    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            pygame.draw.rect(
                screen, (200, 200, 200),
                (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                1
            )

    # Treasure
    pygame.draw.rect(
        screen, (0, 255, 0),
        (TREASURE_STATE[1]*CELL_SIZE, TREASURE_STATE[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Trap
    pygame.draw.rect(
        screen, (255, 0, 0),
        (TRAP_STATE[1]*CELL_SIZE, TRAP_STATE[0]*CELL_SIZE, CELL_SIZE, CELL_SIZE)
    )

    # Agent
    x, y = env.state
    pygame.draw.circle(
        screen, (0, 0, 255),
        (y*CELL_SIZE + CELL_SIZE//2, x*CELL_SIZE + CELL_SIZE//2),
        CELL_SIZE//3
    )

    pygame.display.update()

running = True
while running:
    clock.tick(3)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    action = max(range(4), key=lambda a: agent.q_table[env.state][a])
    _, _, done = env.step(action)

    if done:
        pygame.time.delay(1000)
        env.reset()

    draw()
