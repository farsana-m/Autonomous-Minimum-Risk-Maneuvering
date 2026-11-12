from mrm_environment import SafeStopEnv


env = SafeStopEnv()

observation, info = env.reset()

print()
print("======================================")
print("   AUTONOMOUS SAFE-STOP PROJECT")
print("======================================")
print()
print("MRM Environment: READY")
print()
print("Initial Vehicle State:")
print("--------------------------------------")
print("X Position:", observation[0])
print("Y Position:", observation[1])
print("Speed:", observation[2])
print("Obstacle X:", observation[3])
print("Obstacle Y:", observation[4])
print("Shoulder Distance:", observation[5])
print("--------------------------------------")

for step in range(10):

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = (
        env.step(action)
    )

    print(
        f"Step {step + 1:02d} | "
        f"Reward: {reward:7.2f} | "
        f"Speed: {info['speed']:6.2f} | "
        f"Shoulder Distance: "
        f"{info['shoulder_distance']:6.2f}"
    )

print()
print("Environment test completed successfully.")