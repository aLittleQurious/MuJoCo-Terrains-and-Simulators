import optuna
import numpy as np
import gymnasium as gym
import multiprocessing

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from quadenv import MyQuadRobotEnv

import sys

sys.path.append('/home1/atasever/Robotics/Salamander_Robot/Salamander-Robot-Project')


def make_env(hildebrand_enabled=False,
             spinal_joint_enabled=False, 
             goal_enabled=False,
             xml_file="/home1/atasever/Robotics/Salamander_Robot/Salamander-Robot-Project/xml_files/azhang.xml", 
             #render_mode=None,
             forward_reward_weight= 1,
             ctrl_cost_weight= 0.5,
             goal_reward_weight=1,
             healthy_reward=1.0):
   
    env = MyQuadRobotEnv(
        hildebrand_enabled=hildebrand_enabled,
        spinal_joint_enabled=spinal_joint_enabled,
        goal_enabled=goal_enabled,
        xml_file=xml_file, 
        #render_mode=render_mode,
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        goal_reward_weight=goal_reward_weight,
        healthy_reward=healthy_reward,
    )

    return Monitor(env, filename="monitor.csv")  # Monitor is helpful for recording stats

def optimize_ppo(trial):
    """
    Objective function for Optuna. It:
    1) Samples hyperparams from the trial object.
    2) Creates a PPO model using those hyperparams.
    3) Trains the model for a certain number of timesteps.
    4) Evaluates the model on the environment.
    5) Returns the mean reward (or negative of error) as the objective.
    """

    # ---- Hyperparam search space ----
    n_steps = trial.suggest_int("n_steps", 1000, 3000, step=1000)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    gamma = trial.suggest_float("gamma", 0.90, 0.9999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4, step=0.05)
    forward_reward_weight=trial.suggest_float("forward_reward_weight", 0.4, 2, step=0.4)
    ctrl_cost_weight=trial.suggest_float("ctrl_cost_weight", 0.1, 1, step=0.1)
    goal_reward_weight=trial.suggest_float("goal_reward_weight",  0.4, 2.8, step=0.4)
    healthy_reward=trial.suggest_float("healthy_reward", 0.5, 2.0, step=0.3)

    env = make_env(
        forward_reward_weight=forward_reward_weight,
        ctrl_cost_weight=ctrl_cost_weight,
        goal_reward_weight=goal_reward_weight,
        healthy_reward=healthy_reward)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=n_steps,
        batch_size=batch_size,
        gamma=gamma,
        learning_rate=learning_rate,
        ent_coef=ent_coef,
        clip_range=clip_range,
        verbose=0,  # set to 1 or 2 for more logging
        device="cuda",
    )

    time_steps = 50000
    model.learn(total_timesteps=time_steps)

    # Evaluate the model (returns mean_reward, std_reward)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5, deterministic=True)

    env.close()

    return mean_reward

if __name__ == "__main__":
    # Optional: run parallel studies (helpful if you have multiple cores)
    n_cpu = multiprocessing.cpu_count()

    study = optuna.create_study(direction="maximize")  
    # For reproducibility, you can also set a seed:
    # study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

    study.optimize(optimize_ppo, n_trials=10, n_jobs=min(n_cpu, 4))  # Adjust n_trials and n_jobs as needed

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
