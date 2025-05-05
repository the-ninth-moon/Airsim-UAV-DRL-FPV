import numpy as np
import math

class SimulationEnvironment:

    def __init__(self, drone_position=None, gate_positions=None):
        if gate_positions is None:
            gate_positions = [[1, 0, 0]]
        if drone_position is None:
            drone_position = [0, 0, 0]
        self.drone_position = np.array(drone_position, dtype=np.float64)
        self.gate_positions = [np.array(pos, dtype=np.float64) for pos in gate_positions]
        self.current_gate_index = 0

        # Drone state variables
        self.velocity = np.array([0.0, 0.0, 0.0])  # [vx, vy, vz]
        self.orientation = np.array([0.0, 0.0, 0.0,0.0])  # [roll, pitch, yaw]

        # Simulation parameters
        self.time_step = 0.1  # seconds
        self.max_velocity = 2.0
        self.done = False

    def reset(self):
        """Reset the environment to the initial state."""
        self.drone_position = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0, 0.0,0.0])
        self.current_gate_index = 0
        self.done = False
        return self._get_observation()

    def step(self, action):
        acceleration, yaw_rate = action

        # Update orientation (yaw only for simplicity)
        self.orientation[2] += yaw_rate * self.time_step

        # Update velocity and position
        direction = np.array([
            math.cos(self.orientation[2]),
            math.sin(self.orientation[2]),
            0.0
        ])
        self.velocity += acceleration * direction * self.time_step
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_velocity:
            self.velocity = self.velocity / speed * self.max_velocity

        self.drone_position += self.velocity * self.time_step

        # Calculate reward
        reward, self.done = self._calculate_reward()

        # Return observation, reward, and status
        observation = self._get_observation()
        info = {"current_gate_index": self.current_gate_index}
        return observation, reward, self.done, info

    def _calculate_reward(self):
        """Calculate the reward based on the drone's state and position relative to the gates."""
        if self.current_gate_index >= len(self.gate_positions):
            return 100.0, True  # All gates passed

        gate_position = self.gate_positions[self.current_gate_index]
        distance = np.linalg.norm(self.drone_position - gate_position)

        if distance < 1.0:  # Gate passed
            self.current_gate_index += 1
            return 50.0, False

        if distance > 20.0:  # Out of bounds
            return -10.0, True

        # Shaping reward: closer to the gate is better
        return -distance * 0.1, False

    def _get_observation(self):
        """Return the current observation of the environment."""
        if self.current_gate_index < len(self.gate_positions):
            next_gate_position = self.gate_positions[self.current_gate_index]
        else:
            next_gate_position = np.array([0.0, 0.0, 0.0])

        observation = {
            "position": self.drone_position,
            "velocity": self.velocity,
            "orientation": self.orientation,
            "next_gate": next_gate_position
        }
        return observation

# Example usage
if __name__ == "__main__":
    gate_positions = [[10, 0, 0], [20, 10, 0], [30, 0, 0]]
    env = SimulationEnvironment(drone_position=[0, 0, 0], gate_positions=gate_positions)

    obs = env.reset()
    print("Initial Observation:", obs)

    for _ in range(50):
        action = [1.0, 0.1]  # Constant acceleration and slight yaw rate
        obs, reward, done, info = env.step(action)
        print(f"Step Reward: {reward}, Done: {done}")
        if done:
            break
