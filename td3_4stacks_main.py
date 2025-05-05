import time
from datetime import datetime
import torch
from stable_baselines3 import TD3
from stable_baselines3 import HerReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from supersuit import frame_stack_v1
from environment.drl_environment import DRLEnvironment


def make_env(log_dir, show=False):
    def _init():
        env = DRLEnvironment(viz_image_cv2=False, observation_type="god", show_img=show)
        env.init_race_environment()
        env = frame_stack_v1(env, 4)
        env = Monitor(env, log_dir)
        return env

    return _init


def main():
    n_eval_episodes = 10
    N_eval_freq = 5000
    train_timesteps = 1000000
    deterministic = True

    exp_name = 'TD3_HER_4stacks_A3_S7P_S_Image'
    load_path = ''#f"models/{exp_name}/last_model"
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"

    env = DummyVecEnv([make_env(log_dir, False)])
    policy_kwargs = dict(
        activation_fn=torch.nn.Tanh,
        net_arch=dict(pi=[256, 256,64], qf=[256, 256,64])
    )

    replay_buffer_kwargs = dict(
        n_sampled_goal=4,
        goal_selection_strategy='future',
    )

    if load_path == "":
        model = TD3(
            "MlpPolicy",
            env,
            verbose=0,
            policy_kwargs=policy_kwargs,
            buffer_size=50000,
            batch_size=256,
            tau=0.005,
            gamma=0.95,
            learning_rate=1e-3,
            tensorboard_log=f"SB3/{run_name}/tensorboard",
            device="cpu"
        )
        print("New TD3+HER model created")
    else:
        model = TD3.load(
            load_path,
            env=env,
            tensorboard_log=f"SB3/{run_name}/tensorboard",
            device="cpu"
        )
        print(f'Loaded model from {load_path}')

    for i in range(train_timesteps // 4300):
        model.learn(
            total_timesteps=2010,
        )
        print("------------Model saved--------------")
        model.save(f"models/{exp_name}/last_model")


def test():
    exp_name = 'TD3_HER_4stacks_A3_S7P_S_Image'
    load_path = f"models/{exp_name}/last_model"
    deterministic = True
    n_eval_episodes = 10
    log_dir = f"logs/SB3/{exp_name}/log_train"

    env = DummyVecEnv([make_env(log_dir, True)])
    model = TD3.load(load_path, env=env, device="cpu")

    print(f'Testing model from {load_path}')
    for episode in range(n_eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _ = env.step(action)
    env.close()


if __name__ == '__main__':
    main()
    # test()