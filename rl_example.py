import gym
from tqdm import tqdm
# Example file showing a basic pygame "game loop"
import pygame
from pygame.rect import Rect
import numpy as np
from numpy.random import rand
from numpy.linalg import norm

from qdnn import QDNNL
# pygame setup
pygame.init()
screen = pygame.display.set_mode((400, 400))
clock = pygame.time.Clock()
running = True

input = np.zeros(16)
input[1] = 1


while running:
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("purple")
    # generate random quantum state
    target_state = [1 if rand() >= 0.5 else 0  for _ in range(4)]
    # draw quantum state
    print(target_state)
    for i in range(len(target_state)):
        pygame.draw.rect(screen, (255 * target_state[i], 255 * target_state[i], 255 * target_state[i]), pygame.Rect(30 + 60 * i, 30, 60, 60))
        # pygame.draw.rect(screen, (0 * target_state[i], 0 * target_state[i], 0 * target_state[i]), pygame.Rect(30 + 60 * i, 30, 60, 60))
        
    
    input_layer = QDNNL(4, 0, 0)
    for i in range(500):
        input_layer.epsilon *= 0.95
        results = input_layer.forward(input)
        input_layer.current_results = results
        loss = input_layer.calculate_loss(results, target_state)
        loss_inputs, loss_parameters = input_layer.backpropogate_error(loss)
        input_layer.update_weights(loss_parameters)
        print(loss, results)
        for j in range(len(results)):
            pygame.draw.rect(screen, (255 * results[j], 255 * results[j], 255 * results[j]), pygame.Rect(30 + 60 * j, 200, 60, 60))
        pygame.display.flip()
        clock.tick(50)  # limits FPS to 60
        
    # flip() the display to put your work on screen


pygame.quit()