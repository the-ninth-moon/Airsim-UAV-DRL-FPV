import time
from datetime import datetime
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from supersuit import frame_stack_v1
from environment.drl_environment_lab import DRLEnvironment

def make_env(log_dir,show=False):
    def _init():
        # env = RelativePosEnv()
        # env = DRLEnvironment(viz_image_cv2=False, observation_type="god",show_img=show)
        env = DRLEnvironment(viz_image_cv2=False, observation_type="raw_image",show_img=show)
        env.init_race_environment()
        # env = frame_stack_v1(env, 4)
        # env = Monitor(env, log_dir)
        return env

    return _init


def main():
    n_eval_episodes = 10
    N_eval_freq = 5000
    train_timesteps = 1000000
    deterministic = True

    # exp_name = 'PPO_4stacks_A3_S10P_Image'
    # exp_name = 'PPO_4stacks_A4Rate_S7P_S'
    exp_name = 'PPO_1stacks_A4Rate_S7P_S_raw_final'
    load_path = f"models/{exp_name}/last_model"
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d_%m_%H")
    run_name = f"{exp_name}"
    log_dir = f"logs/SB3/{run_name}/log_train"


    env = DummyVecEnv([make_env(log_dir,False)])  # Notice make_env without parentheses
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

    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 64], vf=[256, 256, 64]),
        activation_fn=torch.nn.Tanh,
        ortho_init=True,
        log_std_init=-0.6,
    )
    if load_path == "":
        model = PPO(
            "MultiInputPolicy",  # *** 关键修改：使用 MultiInputPolicy 处理 Dict 观察空间 ***
            env,
            verbose=0,
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"SB3/{run_name}/tensorboard",
            gamma=0.95,
            batch_size=50,
            clip_range=0.25,
            n_steps=1000,
            gae_lambda=0.95,
            n_epochs=6,
            ent_coef=0.01,
            learning_rate=3e-4,
            device="auto"
        )
        print("New model is created")
    else:
        model = PPO.load(
            load_path,
            verbose=0,
            tensorboard_log=f"SB3/{run_name}/tensorboard",
            env=env,
            device="auto"
        )
        print('The previous model is loaded from ', load_path)
    # policy_kwargs = dict(activation_fn=torch.nn.Tanh,
    #                      net_arch = dict(pi=[256,256,64], vf=[256,256,64]),
    #                      ortho_init = True,
    #                      log_std_init = -0.6,
    #                     )
    # if load_path == "":
    #     model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs,tensorboard_log=f"SB3/{run_name}/tensorboard",gamma=0.95,batch_size=50
    #                 ,clip_range=0.25,n_steps=1000,gae_lambda=0.95,n_epochs=6,ent_coef=0.01,learning_rate=3e-4,device="cpu")
    #     # model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard")
    #     print("New model is created")
    # else:
    #     model = PPO.load(load_path, verbose=0,tensorboard_log=f"SB3/{run_name}/tensorboard", env=env,device="auto")
    #     # model = RecurrentPPO.load(load_path, verbose=0, tensorboard_log=f"SB3/{run_name}/tensorboard", env=vec_env)
    #     print('The previous model is loaded from ', load_path)
    for i in range(train_timesteps // 4300):
        model.learn(total_timesteps=10100)
        print("------------模型已保存--------------")
        model.save(f"models/{exp_name}/last_model")


def test():
    exp_name = 'PPO_4stacks_A4Rate_S7P_S_final'
    load_path = f"models/{exp_name}/last_model"
    deterministic = True # 测试时通常设置为 True，以保证动作的确定性
    n_eval_episodes = 10 # 测试的回合数
    log_dir = f"logs/SB3/{exp_name}/log_train"
    env = DummyVecEnv([make_env(log_dir,False)]) # 创建测试环境
    policy_kwargs = dict(activation_fn=torch.nn.Tanh,
                         net_arch = dict(pi=[256,256,64], vf=[256,256,64]),
                         ortho_init = True,
                         log_std_init = -0.6,
                        )
    model = PPO.load(load_path, env=env, policy_kwargs=policy_kwargs, device="cpu") # 加载模型，并传入环境和 policy_kwargs
    print(f'The model is loaded from {load_path}')

    total_reward = 0
    for episode in range(n_eval_episodes):
        obs = env.reset()
        episode_reward = 0
        done = False
        step_num = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic) # 使用模型预测动作，deterministic=True 保证动作确定性
            obs, reward, done, info = env.step(action) # 环境执行动作
            episode_reward += reward
            step_num += 1
            # if done:
            #     print(f"Episode {episode+1}/{n_eval_episodes}, Episode Reward: {episode_reward:.2f}, Episode Length: {step_num}")
            #     total_reward += episode_reward
            #     break

    # avg_reward = total_reward / n_eval_episodes
    # print(f"-----------------------------------")
    # print(f"Average Reward over {n_eval_episodes} episodes: {avg_reward:.2f}")
    # print(f"-----------------------------------")
    env.close() # 关闭环境

if __name__ == '__main__':
    # test()
    main()