import time
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO,SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.torch_layers import NatureCNN  # CNN from Nature paper
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from environment import drl_environment
import wandb
from wandb.integration.sb3 import WandbCallback

from environment.drl_environment import DRLEnvironment

def make_env(log_dir):
    def _init():
        env = DRLEnvironment(viz_image_cv2=False, observation_type="god")
        env.init_race_environment()
        env = Monitor(env, log_dir)
        return env

    return _init


def main():
    n_eval_episodes = 8
    N_eval_freq = 500
    train_timesteps = 1000000
    deterministic = True


    exp_name = 'PPO_lstm_0215A3S19P_Easy_Image'
    load_path =f"models/{exp_name}/last_model"
    wandb_project_name = "Airsim_Drone_Gate"
    wandb_entity = "onurakgun"
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DummyVecEnv([make_env(log_dir)])  # Notice make_env without parentheses
    # eval_callback = EvalCallback(
    #     env,
    #     best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
    #     log_path=f"SB3/{run_name}/logs",
    #     eval_freq=N_eval_freq,  # 评估频率
    #     n_eval_episodes=n_eval_episodes,  # 评估回合数
    #     deterministic=deterministic,
    #     render=True,
    #     verbose=True
    # )
    if load_path == "":
        # model = PPO("MultiInputPolicy", env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard",gamma=0.92,batch_size=20
        #             ,clip_range=0.25,n_steps=200,gae_lambda=0.95,n_epochs=30,ent_coef=0.02,learning_rate=3e-4,device="cuda")
        model = RecurrentPPO("MultiInputLstmPolicy",
                             env,
                             verbose=0,
                             tensorboard_log=f"SB3/{run_name}/tensorboard",
                             gamma=0.92,
                             batch_size=64,
                             clip_range=0.25,
                             n_steps=64,
                             gae_lambda=0.95,
                             n_epochs=8,
                             ent_coef=0.02,
                             learning_rate=3e-4,
                             device="cuda")
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        print("New model is created")
    else:
        # model = PPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=env,gamma=0.92,batch_size=50
        #             ,clip_range=0.25,n_steps=200,gae_lambda=0.95,n_epochs=30,ent_coef=0.02,learning_rate=3e-4,device="cuda")
        model = RecurrentPPO.load(load_path,
                             env=env,
                             verbose=0,
                             tensorboard_log=f"SB3/{run_name}/tensorboard",
                              gamma=0.92,
                              batch_size=64,
                              clip_range=0.25,
                              n_steps=64,
                              gae_lambda=0.95,
                              n_epochs=8,
                              ent_coef=0.02,
                              learning_rate=3e-4,
                              device="cuda")
        # model = RecurrentPPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=vec_env)
        print('The previous model is loaded from ', load_path)

    for i in range(train_timesteps // 1000):
        model.learn(total_timesteps=1000)
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")


if __name__ == '__main__':
    main()