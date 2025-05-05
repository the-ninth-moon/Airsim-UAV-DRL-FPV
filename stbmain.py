import time
from datetime import datetime
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from stable_baselines3 import A2C, PPO, SAC, TD3
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
        env = DRLEnvironment(viz_image_cv2=False, observation_type="god",show_img=False)
        env.init_race_environment()
        return env

    return _init


def main():
    n_eval_episodes = 8
    N_eval_freq = 500
    train_timesteps = 1000000
    deterministic = True


    exp_name = 'PPO_A3_S6P_S_Image'
    load_path = ''#f"models/{exp_name}/last_model"
    wandb_project_name = "Airsim_Drone_Gate"
    wandb_entity = "onurakgun"
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DRLEnvironment(viz_image_cv2=False, observation_type="god", show_img=False)
    env.init_race_environment()
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=dict(pi=[128,256,128,64], vf=[128,256,128,64]))
    if load_path == "":
        model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs,tensorboard_log=f"SB3/{run_name}/tensorboard",gamma=0.95,batch_size=100
                    ,clip_range=0.25,n_steps=1000,gae_lambda=0.95,n_epochs=5,ent_coef=0.01,learning_rate=3e-4,device="cpu")
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        print("New model is created")
    else:
        model = PPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=env,gamma=0.95,batch_size=50
                    ,clip_range=0.25,n_steps=1000,gae_lambda=0.95,n_epochs=8
                         ,ent_coef=0.01,learning_rate=3e-4,device="cpu")
        # model = RecurrentPPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=vec_env)
        print('The previous model is loaded from ', load_path)
    for i in range(train_timesteps // 5000):
        model.learn(total_timesteps=5000)
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")

def main_lstm_ppo():
    n_eval_episodes = 8
    N_eval_freq = 500
    train_timesteps = 1000000
    deterministic = True


    exp_name = 'PPO_lstm_0228_A3_S14P_Easy_Image'
    load_path = f"models/{exp_name}/last_model"
    wandb_project_name = "Airsim_Drone_Gate"
    wandb_entity = "onurakgun"
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DummyVecEnv([make_env(log_dir)])  # Notice make_env without parentheses
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
        log_path=f"SB3/{run_name}/logs",
        eval_freq=N_eval_freq,  # 评估频率
        n_eval_episodes=n_eval_episodes,  # 评估回合数
        deterministic=deterministic,
        render=True,
        verbose=True
    )
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=dict(pi=[128,128,64], vf=[128,128,64]))
    if load_path == "":
        model = RecurrentPPO("MultiInputLstmPolicy",
                             env=env,
                             verbose=0,
                             policy_kwargs=policy_kwargs,
                             tensorboard_log=f"SB3/{run_name}/tensorboard",
                              gamma=0.95,
                              batch_size=512,
                              clip_range=0.25,
                              n_steps=512,
                              gae_lambda=0.95,
                              n_epochs=10,
                              ent_coef=0.02,
                              learning_rate=3e-4,
                              device="cuda")
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        print("New model is created")
    else:
        model = RecurrentPPO.load(load_path,
                             env=env,
                             verbose=0,
                             tensorboard_log=f"SB3/{run_name}/tensorboard",
                              gamma=0.95,
                            policy_kwargs=policy_kwargs,
                              batch_size=512,
                              clip_range=0.25,
                              n_steps=512,
                              gae_lambda=0.95,
                              n_epochs=30,
                              ent_coef=0.02,
                              learning_rate=3e-4,
                              device="cuda")
        print('The previous model is loaded from ', load_path)
    for i in range(train_timesteps // 5000):
        model.learn(total_timesteps=5000)
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")

def main_td3():
    n_eval_episodes = 8
    N_eval_freq = 500
    train_timesteps = 1000000
    deterministic = True

    exp_name = 'TD3_0228_A3_S11P_Easy_Image' # 修改为TD3的实验名称
    load_path = "" # TD3 通常从头开始训练，如果需要加载，请修改路径
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DummyVecEnv([make_env(log_dir)])  # 保持环境创建方式

    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
        log_path=f"SB3/{run_name}/logs",
        eval_freq=N_eval_freq,  # 评估频率
        n_eval_episodes=n_eval_episodes,  # 评估回合数
        deterministic=deterministic,
        render=False, # TD3 训练时通常不需要render，评估时可以打开
        verbose=True
    )

    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=[256, 256,128]) # TD3 通常使用更深的网络，这里简化为两层256
    if load_path == "":
        model = TD3("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs, tensorboard_log=f"SB3/{run_name}/tensorboard",
                    gamma=0.92,  # TD3 常用较高的 gamma
                    learning_rate=1e-3, # TD3 常用较小的学习率
                    buffer_size=50000, # TD3 需要较大的 replay buffer
                    learning_starts=100, # 学习开始前的探索步数
                    batch_size=100, # TD3 常用 batch size
                    tau=0.005, # 目标网络软更新系数
                    # 策略噪声
                    target_policy_noise=0.4, # 目标策略平滑噪声
                    target_noise_clip=0.5, # 目标噪声裁剪
                    device="cuda")
        print("New TD3 model is created")
    else:
        model = TD3.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=env,
                    gamma=0.92,  # TD3 常用较高的 gamma
                    learning_rate=1e-3, # TD3 常用较小的学习率
                    buffer_size=50000, # TD3 需要较大的 replay buffer
                    learning_starts=100, # 学习开始前的探索步数
                    batch_size=100, # TD3 常用 batch size
                    tau=0.005, # 目标网络软更新系数
                    target_policy_noise=0.4, # 目标策略平滑噪声
                    target_noise_clip=0.5, # 目标噪声裁剪
                    device="cuda")
        print('The previous TD3 model is loaded from ', load_path)

    for i in range(train_timesteps // 5000):
        model.learn(total_timesteps=5000, callback=eval_callback) # 添加 eval_callback
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")

def main_SAC():
    n_eval_episodes = 8
    N_eval_freq = 500
    train_timesteps = 1000000
    deterministic = True

    exp_name = 'SAC_0303_A3_Discrete_S15P_Easy_Image'  # 修改了实验名称以反映算法
    load_path = '' #f"models/{exp_name}/last_model"
    wandb_project_name = "Airsim_Drone_Gate"
    wandb_entity = "onurakgun"
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DummyVecEnv([make_env(log_dir)])  # 保持环境创建方式不变
    eval_callback = EvalCallback(
        env,
        best_model_save_path=f"SB3/{run_name}/saved_model/best_model",
        log_path=f"SB3/{run_name}/logs",
        eval_freq=N_eval_freq,  # 评估频率
        n_eval_episodes=n_eval_episodes,  # 评估回合数
        deterministic=deterministic,
        render=True,
        verbose=True
    )
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch=dict(pi=[128,128,64], qf=[128,128,64])) # SAC使用qf代替vf
    if load_path == "":
        model = SAC("MultiInputPolicy", env, verbose=0, policy_kwargs=policy_kwargs,tensorboard_log=f"SB3/{run_name}/tensorboard",gamma=0.92,batch_size=200
                    ,ent_coef='auto_0.01',learning_rate=6e-4,device="cuda", buffer_size=200000, learning_starts=1000) # 替换为SAC, 并调整SAC相关参数
        # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
        print("New SAC model is created")
    else:
        model = SAC.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=env,gamma=0.92,batch_size=200
                    ,ent_coef='auto_0.01',learning_rate=6e-4,device="cuda", buffer_size=200000, learning_starts=1000) # 替换为SAC.load，并调整SAC相关参数
        # model = RecurrentPPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=vec_env)
        print('The previous SAC model is loaded from ', load_path)
    for i in range(train_timesteps // 5000):
        model.learn(total_timesteps=5000, callback=eval_callback) # 可以加入eval_callback以便在训练中评估
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")

if __name__ == '__main__':
    main()