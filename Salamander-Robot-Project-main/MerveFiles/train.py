import gymnasium as gym
from stable_baselines3 import SAC, TD3, A2C, PPO, DDPG
import os
import argparse
from datetime import date

from pathlib import Path
from quadenv import MyQuadRobotEnv


model_dir = "Models"
log_dir_init = "Logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir_init, exist_ok=True)

today = str(date.today())

env = MyQuadRobotEnv(hildebrand_enabled=False,
                 spinal_joint_enabled=False, 
                 goal_enabled=False,
                 xml_file="xml_files/azhang.xml", 
                 frame_skip=5,
                 include_cfrc_ext_in_observation=False,
                 forward_reward_weight= 1,
                 ctrl_cost_weight= 0.5,
                 contact_cost_weight=5e-4,
                 goal_reward_weight=1,
                 healthy_reward=1.0,
                 terminate_when_unhealthy=True,
                 healthy_z_range=(0.07, 0.09),
                 contact_force_range=(-1.0, 1.0),
                 reset_noise_scale=0.1,
                 exclude_current_positions_from_observation=False) 

def train(env, sb3_algo):

    log_dir = f"{log_dir_init}/{sb3_algo}-{today}"
    
    if sb3_algo == 'SAC':
        model = SAC('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir,learning_rate=0.01)
    elif sb3_algo == 'TD3':
        model = TD3('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'DQN':
        # DQN works on discrete action space
        model = DQN('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    elif sb3_algo == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device='cpu', tensorboard_log=log_dir)
    else:
        print('Algorithm not found!')
        return

    TIMESTEPS = 10000
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
        model.save(f"{model_dir}/{sb3_algo}_{TIMESTEPS * iters}")


def test(env, sb3_algo, path_to_model):
    if sb3_algo == 'SAC':
        model = SAC.load(path_to_model, env=env)
    elif sb3_algo == 'TD3':
        model = TD3.load(path_to_model, env=env)
    elif sb3_algo == 'A2C':
        model = A2C.load(path_to_model, env=env)
    elif sb3_algo == 'DQN':
        # DQN works on discrete action space
        model = DQN.load(path_to_model, env=env)
    elif sb3_algo == 'PPO':
        model = PPO.load(path_to_model, env=env)
    else:
        print('Algorithm not found!')
        return

    # First observation in the env
    obs = env.reset()[0]
    done = False
    extra_steps = 500
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done:
            extra_steps -= 1

            if extra_steps < 0:
                break


if __name__ == '__main__':

    # Parse command line inputs

    parser = argparse.ArgumentParser(description='Train or Test model.')
    parser.add_argument('gymenv', help='Gymnasium environment i.e Humanoid-v4')
    parser.add_argument('sb3_algo', help='StableBaseline3 RL Algo i.e SAC TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    if args.train:
        gymenv = gym.make(args.gymenv, render_mode=None)
        train(gymenv, args.sb3_algo)

    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make(args.gymenv, render_mode='human')
            test(gymenv, args.sb3_algo, path_to_model=args.test)

        else:

            print(f'{args.test} not found')
