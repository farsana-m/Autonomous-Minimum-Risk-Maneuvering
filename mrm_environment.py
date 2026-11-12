import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SafeStopEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self):

        super().__init__()

        # Action:
        # [steering, acceleration/braking]

        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        # State:
        # x position
        # y position
        # speed
        # obstacle x
        # obstacle y
        # distance to shoulder

        self.observation_space = spaces.Box(
            low=np.array([
                0.0,
                -5.0,
                0.0,
                0.0,
                -5.0,
                0.0
            ], dtype=np.float32),

            high=np.array([
                200.0,
                5.0,
                40.0,
                200.0,
                5.0,
                10.0
            ], dtype=np.float32)
        )

        self.dt = 0.1
        self.max_steps = 300

        self.shoulder_y = -3.5

        self.x = 0.0
        self.y = 0.0
        self.speed = 20.0

        self.obstacle_x = 80.0
        self.obstacle_y = 0.0

        self.step_count = 0

    # -----------------------------------------------------
    # RESET
    # -----------------------------------------------------

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.step_count = 0

        self.x = 0.0
        self.y = 0.0
        self.speed = 20.0

        # Keep obstacle far enough for the agent
        # to perform the maneuver.

        self.obstacle_x = self.np_random.uniform(
            60.0,
            100.0
        )

        self.obstacle_y = 0.0

        return self._get_observation(), {}

    # -----------------------------------------------------
    # STEP
    # -----------------------------------------------------

    def step(self, action):

        steering = float(action[0])
        acceleration = float(action[1])

        self.step_count += 1

        steering = np.clip(
            steering,
            -1.0,
            1.0
        )

        acceleration = np.clip(
            acceleration,
            -1.0,
            1.0
        )

        # -------------------------------------------------
        # VEHICLE DYNAMICS
        # -------------------------------------------------

        # Stronger braking response than previous version.

        self.speed += (
            acceleration
            * 4.0
            * self.dt
        )

        self.speed = np.clip(
            self.speed,
            0.0,
            30.0
        )

        self.y += (
            steering
            * 2.5
            * self.dt
        )

        self.y = np.clip(
            self.y,
            -4.5,
            4.5
        )

        self.x += (
            self.speed
            * self.dt
        )

        # -------------------------------------------------
        # DISTANCES
        # -------------------------------------------------

        shoulder_distance = abs(
            self.y - self.shoulder_y
        )

        longitudinal_distance = (
            self.obstacle_x - self.x
        )

        lateral_distance = abs(
            self.obstacle_y - self.y
        )

        # -------------------------------------------------
        # COLLISION
        # -------------------------------------------------

        # Collision only if vehicle is actually in
        # the obstacle's lane region.

        collision = (
            longitudinal_distance >= -2.0
            and longitudinal_distance <= 5.0
            and lateral_distance < 1.5
        )

        # -------------------------------------------------
        # REWARD
        # -------------------------------------------------

        reward = 0.0

        # 1. Strong reward for reaching shoulder

        shoulder_progress = (
            3.5 - shoulder_distance
        )

        reward += (
            2.0 * shoulder_progress
        )

        # 2. Reward low speed during emergency stop

        if self.x > 20.0:

            reward += (
                max(0.0, 12.0 - self.speed)
                * 0.8
            )

        # 3. Reward controlled forward progress

        reward += 0.03 * self.speed

        # 4. Penalize aggressive steering

        reward -= (
            0.3
            * abs(steering)
        )

        # 5. Penalize aggressive acceleration

        reward -= (
            0.1
            * abs(acceleration)
        )

        # 6. Encourage slowing near obstacle

        if (
            0.0 < longitudinal_distance < 30.0
        ):

            reward += (
                max(0.0, 20.0 - self.speed)
                * 0.5
            )

        # 7. Large collision penalty

        if collision:

            reward -= 150.0

        # -------------------------------------------------
        # SAFE STOP
        # -------------------------------------------------

        safe_stop = (
            shoulder_distance <= 1.1
            and self.speed < 5.0
        )

        if safe_stop:

            reward += 200.0

        # -------------------------------------------------
        # TERMINATION
        # -------------------------------------------------

        terminated = (
            collision
            or safe_stop
        )

        truncated = (
            self.step_count >= self.max_steps
        )

        observation = self._get_observation()

        info = {

            "collision": collision,

            "safe_stop": safe_stop,

            "speed": self.speed,

            "lateral_position": self.y,

            "shoulder_distance":
                shoulder_distance,

            "obstacle_distance":
                abs(longitudinal_distance),

            "longitudinal_distance":
                longitudinal_distance
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    # -----------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------

    def _get_observation(self):

        shoulder_distance = abs(
            self.y - self.shoulder_y
        )

        return np.array([

            self.x,

            self.y,

            self.speed,

            self.obstacle_x,

            self.obstacle_y,

            shoulder_distance

        ], dtype=np.float32)