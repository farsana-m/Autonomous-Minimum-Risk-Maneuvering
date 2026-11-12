from stable_baselines3 import PPO
from mrm_environment import SafeStopEnv
import numpy as np


print("==============================================")
print("     AUTONOMOUS SAFE-STOP - DRL EVALUATION")
print("==============================================")

# Create environment
env = SafeStopEnv()

# Load trained PPO model
model = PPO.load("models/mrm_ppo")

print("\nTrained PPO model loaded successfully.")
print("Starting evaluation...\n")


total_episodes = 10
safe_stops = 0
collisions = 0
timeouts = 0

for episode in range(total_episodes):

    observation, info = env.reset()

    done = False
    total_reward = 0.0
    steps = 0

    while not done:

        # Ask trained PPO agent for an action
        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward
        steps += 1

        done = terminated or truncated

    if info["safe_stop"]:
        safe_stops += 1
        result = "SAFE STOP"

    elif info["collision"]:
        collisions += 1
        result = "COLLISION"

    else:
        timeouts += 1
        result = "TIMEOUT"

    print(
        f"Episode {episode + 1:02d} | "
        f"{result:12s} | "
        f"Reward: {total_reward:8.2f} | "
        f"Steps: {steps:3d} | "
        f"Final Speed: {info['speed']:6.2f} | "
        f"Shoulder Distance: "
        f"{info['shoulder_distance']:5.2f}"
    )


print("\n==============================================")
print("              EVALUATION RESULTS")
print("==============================================")

print(f"Total Episodes : {total_episodes}")
print(f"Safe Stops     : {safe_stops}")
print(f"Collisions     : {collisions}")
print(f"Timeouts       : {timeouts}")

success_rate = (
    safe_stops / total_episodes
) * 100

print(f"Safe-Stop Rate : {success_rate:.1f}%")

print("==============================================")