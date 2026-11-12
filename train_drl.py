from stable_baselines3 import PPO
from mrm_environment import SafeStopEnv


print("==========================================")
print("   AUTONOMOUS SAFE-STOP - PPO TRAINING")
print("==========================================")

# Create environment
env = SafeStopEnv()

print("\nMRM environment created.")
print("Starting PPO training...\n")

# PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    gamma=0.99,
    gae_lambda=0.95,
)

# Train
model.learn(total_timesteps=50000)

# Save trained model
model.save("models/mrm_ppo")

print("\n==========================================")
print("          TRAINING COMPLETED")
print("==========================================")
print("Model saved to:")
print("models/mrm_ppo")