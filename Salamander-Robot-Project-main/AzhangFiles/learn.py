import gymnasium as gym
from stable_baselines3 import PPO, SAC
import os
from ball_balance_env import BallBalanceEnv

# We may train multiple models of same family (PPO, DDPG, ...). This variable adds the number to models name so we can distict them
model_number = 1
model_name = "SAC"

# Directory of the models and logs 
models_dir = f"models/{model_name}-{model_number}"
log_dir = f"logs"

# if the directory is not available we will create them here 
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(log_dir):
    os.makedirs(log_dir)


# create and reset the environment
env = BallBalanceEnv(render_mode="rgb_array")
env.reset()


# The RL model is defined here, only change the name of the function (here A2C), hyper parameters can also be changed here
model = SAC("MlpPolicy", env, verbose = 1, tensorboard_log=log_dir)

# we will save models based on TIMESTEPS variable
TIMESTEPS = 10000

# we will let the model learn for "n * TIMESTEPS" steps. by changing the n in the for loop below we can change the duration of learning
for i in range(1,40):
    model.learn(total_timesteps= TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"{model_name}-{model_number}" )
    model.save(f"{models_dir}/{TIMESTEPS*i}")