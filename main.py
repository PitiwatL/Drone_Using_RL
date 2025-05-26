import pygame
from drone import Drone
from pygame import Rect
import numpy as np
import math
from typing import Tuple
from flight_controller import FlightController
from policy_grad import PolicyNet2Layer

#---------------------WRITE YOUR OWN CODE HERE------------------------#
from heuristic_controller import HeuristicController
from custom_controller import CustomController

def generate_controller() -> FlightController:
    return HeuristicController() # <--- Replace this with your own written controller
    # return CustomController()

def is_training() -> bool:
    return False # <--- Replace this with True if you want to train, false otherwise
def is_saving() -> bool:
    return False # <--- Replace this with True if you want to save the results of training, false otherwise

#---------------------------------------------------------------------#
SCREEN_WIDTH = 720
SCREEN_HEIGHT = 480


def get_scale():
    return min(SCREEN_HEIGHT, SCREEN_WIDTH)

def convert_to_screen_coordinate(x,y):
    scale = get_scale()
    return (x*scale + SCREEN_WIDTH/2, -y*scale + SCREEN_HEIGHT/2)

def convert_to_screen_size(game_size):    
    scale = get_scale()
    return game_size*scale

def convert_to_game_coordinates(x,y):
    scale = get_scale()
    return ((x - SCREEN_WIDTH/2)/scale, (y - SCREEN_HEIGHT/2)/scale)

def status(controller: FlightController) :
    pass



def main(controller: HeuristicController):

    # Initialise pygame
    pygame.init()
    clock = pygame.time.Clock()

    # Load the relevant graphics into pygame
    drone_img = pygame.image.load('graphics/drone_small.png')
    background_img = pygame.image.load('graphics/background.png')
    target_img = pygame.image.load('graphics/target.png')

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    
    # Initalise the drone
    heuristic = HeuristicController()
    drone = heuristic.init_drone(random = True)

    simulation_step_counter = 0
    max_simulation_steps = heuristic.get_max_simulation_steps()

    delta_time = heuristic.get_time_interval()

    # model = PolicyNet2Layer()
    states, actions, rewards = [], [], []
    running = True
    num_episode = 0
    count_reward = 0
    hit_target_count = 0
    load_weights = True
    if load_weights == True:
        heuristic.load_weights("drone_episode_1800_8801.7286.npz")
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # --- Begin Physics --- #
        # Get the thrust information from the controller
        # drone.set_thrust(controller.get_thrusts_heuristics(drone))

        action, tuple_state, reached_target = heuristic.train(drone=drone, train = False)

        drone.set_thrust(action)

        # collect states, actions, values
        states.append(tuple_state[0]) 
        actions.append(tuple_state[1])
        rewards.append(tuple_state[2])
        
        # Update the simulation
        drone.step_simulation(delta_time)

        # --- Begin Drawing --- #

        # Refresh the background
        screen.blit(background_img, (0,0))
        # Draw the current drone on the screen
        draw_drone(screen, drone, drone_img)
        # Draw the next target on the screen
        draw_target(drone.get_next_target(), screen, target_img)

        # Actually displays the final frame on the screen
        pygame.display.flip()

        # speed up the simulation
        clock.tick(80)

        # Checks whether to reset the current drone
        simulation_step_counter+=1    
        if (simulation_step_counter>=max_simulation_steps) or reached_target or abs(drone.x) > 0.7 or abs(drone.y) > 0.7: 
            num_episode += 1
            # print("Num Episode: ", num_episode)
            if reached_target :
                hit_target_count +=1 

            # sum_reward = heuristic.update_param(num_episode, states, actions, rewards)
            # count_reward += sum_reward

            # Reset the drone
            states, actions, rewards = [], [], []
            drone = heuristic.init_drone(random = True) 
            simulation_step_counter = 0

            if num_episode % 300 == 0 :
                # print(count_reward)
                # heuristic.save_weights(filename = f"drone_episode_{num_episode}_{count_reward:.4f}.npz")
                count_reward = 0
                print("Accuracy: ", hit_target_count/num_episode)
                print("Save complete!")


def draw_target(target_point, screen, target_img):
    target_size = convert_to_screen_size(0.1)
    point_x, point_y = convert_to_screen_coordinate(*target_point)
    screen.blit(pygame.transform.scale(target_img, (int(target_size), int(target_size))), (point_x-target_size/2, point_y-target_size/2))

def draw_drone(screen: pygame.Surface, drone: Drone, drone_img: pygame.Surface):
    drone_x, drone_y = convert_to_screen_coordinate(drone.x, drone.y)
    drone_width = convert_to_screen_size(0.3)
    drone_height = convert_to_screen_size(0.15)
    drone_rect = Rect(drone_x-drone_width/2, drone_y-drone_height/2, drone_width, drone_height)
    drone_scaled_img = pygame.transform.scale(drone_img, (int(drone_width), int(drone_height)))
    drone_scaled_center = drone_scaled_img.get_rect(topleft = (drone_x-drone_width/2, drone_y-drone_height/2)).center
    rotated_drone_img = pygame.transform.rotate(drone_scaled_img, -drone.pitch * 180 / math.pi)
    drone_scaled_rect = rotated_drone_img.get_rect(center=drone_scaled_center)
    screen.blit(rotated_drone_img, drone_scaled_rect)

if __name__ == "__main__":
    controller = generate_controller()
    if is_training():
        controller.train()
        if is_saving():
            controller.save()        
    else:
        controller.load()
    
    main(controller)