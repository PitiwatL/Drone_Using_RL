import numpy as np
from flight_controller import FlightController
from drone import Drone
from typing import Tuple
from policy_grad import PolicyNet2Layer

class HeuristicController(FlightController):
    def __init__(self):
        """
         Creates a heuristic flight controller with some specified parameters
        """
        self.ky = 1.0
        self.kx = 0.5
        self.abs_pitch_delta = 0.1
        self.abs_thrust_delta = 0.3
        self.policy = PolicyNet2Layer()

    def init_drone(self, random = True) -> Drone:
        """Creates a Drone object initialised with a deterministic set of target coordinates.

        Returns:
            Drone: An initial drone object with some programmed target coordinates.
        """

        # This is used to set up the coordinate
        drone = Drone()

        if random == False: 
            drone.add_target_coordinate((0.4, 0.4))
            drone.add_target_coordinate((-0.4, 0.4))
            drone.add_target_coordinate((-0.4, -0.4))
            drone.add_target_coordinate((-0.4, 0.4))
        
        if random: 
            for _ in range(10000):
                x_target, y_target = np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5)
                drone.add_target_coordinate((x_target, y_target))
                # drone.add_target_coordinate((0.4, 0.4))
                # drone.add_target_coordinate((-0.4, 0.4))
                # drone.add_target_coordinate((-0.4, -0.4))
                # drone.add_target_coordinate((-0.4, 0.4))

        return drone
    
    def get_max_simulation_steps(self):
        return 400 
    # You can alter the amount of steps you want your program to run for here


    def get_thrusts_heuristics(self, drone: Drone) -> Tuple[float, float]:
        """Takes a given drone object, containing information about its current state
        and calculates a pair of thrust values for the left and right propellers.

        Args:
            drone (Drone): The drone object containing the information about the drones state.

        Returns:
            Tuple[float, float]: A pair of floating point values which respectively represent the thrust of the left and right propellers, must be between 0 and 1 inclusive.
        """

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y

        thrust_adj = np.clip(dy * self.ky, -self.abs_thrust_delta, self.abs_thrust_delta)
        target_pitch = np.clip(dx * self.kx, -self.abs_pitch_delta, self.abs_pitch_delta)
        delta_pitch = target_pitch-drone.pitch

        thrust_left = np.clip(0.5 + thrust_adj + delta_pitch, 0.0, 1.0)
        thrust_right = np.clip(0.5 + thrust_adj - delta_pitch, 0.0, 1.0)

        # The default controller sets each propeller to a value of 0.5 0.5 to stay stationary.
        return (thrust_left, thrust_right)

    def train(self, drone: Drone, reached_target = None, train = True):
        """A self contained method designed to train parameters created in the initialiser.
        """
        # --- Code snipped provided for guidance only --- #
        # for n in range(epochs):
        #     # 1) modify parameters
            
        #     # 2) create a new drone simulation
        #     drone = self.init_drone()
        #     # 3) run simulation
        #     for t in range(self.get_max_simulation_steps()):
        #         drone.set_thrust(self.get_thrusts(drone))
        #         drone.step_simulation(self.get_time_interval())
        #     # 4) measure change in quality

        #     # 5) update parameters according to algorithm

        target_point = drone.get_next_target()
        dx = target_point[0] - drone.x
        dy = target_point[1] - drone.y
        vx = drone.velocity_x
        vy = drone.velocity_y

        dist_to_target = (dx**2 + dy**2)**0.5

        state = np.array([drone.x, drone.y,
                          target_point[0], target_point[1]
                ])
        # print(state)
        # print(state)
        mu, sigma = self.policy.forward(state, train = train)

        action = self.policy.sample_action()
        
        reward = -dist_to_target/50
        reached_target = False
        if dist_to_target < 0.17 :  #reached_target :
            print("Hit the target")
            reached_target = True
            reward = 40
        
        # reward = np.clip(reward, -10, 20)


        # grads = self.policy.compute_gradients(action, return_R=reward)
        # self.policy.update(grads)

        # thrust_left, thrust_right = action.tolist()[0], action.tolist()[1]
        thrust_left = np.clip(action.tolist()[0], 0.0, 1)
        thrust_right = np.clip(action.tolist()[1], 0.0, 1)
     
        # print("loss: ", self.policy.log_prob(action))
        # print(dist_to_target)
        return (thrust_left, thrust_right), (state, action, reward), reached_target
    
    def update_param(self, num_episode, states, actions, rewards):
        gamma = 0.95
        returns = []
        R = 0
        for r in rewards:
            R = r + gamma * R
            returns.append(R)

        # returns = np.array(returns)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Initialize accumulators for gradients
        total_dW1 = np.zeros_like(self.policy.W1)
        total_db1 = np.zeros_like(self.policy.b1)
        total_dW2 = np.zeros_like(self.policy.W2)
        total_db2 = np.zeros_like(self.policy.b2)
        total_dW3 = np.zeros_like(self.policy.W3)
        total_db3 = np.zeros_like(self.policy.b3)

        # Accumulate gradients for the whole episode
        for s, a, accu_reward in zip(states, actions, returns):
            self.policy.forward(s)  # Update internal activations
            grads = self.policy.compute_gradients(a)

            dW1, db1, dW2, db2, dW3, db3 = grads

            total_dW1 += dW1 * sum(returns)
            total_db1 += db1 * sum(returns)
            total_dW2 += dW2 * sum(returns)
            total_db2 += db2 * sum(returns)
            total_dW3 += dW3 * sum(returns)
            total_db3 += db3 * sum(returns)

        # print(total_dW1)
        # Apply policy gradient update
        self.policy.update((total_dW1, total_db1, total_dW2, total_db2, total_dW3, total_db3), lr=1e-5)

        print(f"Episode {num_episode}, Total Reward: {sum(returns)}")
        return sum(returns)

    def save_weights(self, filename) :
         self.policy.save_weights(filename= filename)
    
    def load_weights(self, filename) :
         self.policy.load_weights(filename= filename)

    def load(self):
        """Load the parameters of this flight controller from disk.
        """
        try:
            parameter_array = np.load('heuristic_controller_parameters.npy')
            self.ky = parameter_array[0]
            self.kx = parameter_array[1]
            self.abs_pitch_delta = parameter_array[2]
            self.abs_thrust_delta = parameter_array[3]
        except:
            print("Could not load parameters, sticking with default parameters.")

    def save(self):
        """Save the parameters of this flight controller to disk.
        """
        parameter_array = np.array([self.ky, self.kx, self.abs_pitch_delta, self.abs_thrust_delta])
        np.save('heuristic_controller_parameters.npy', parameter_array)
        