import os
import glob
import time
from datetime import datetime
from argparse import ArgumentParser
import pandas as pd
import matplotlib.pyplot as plt

import torch
import numpy as np

# from agents.PPO import PPO
from agents.SAC_normal_sample import SAC
from agents.ddpg_brain import DDPGAgent
from agents.PPO import PPO
from environment.drl_environment import DRLEnvironment
from agents.PPO_linear import PPO_linear

def train():
    env_name = "DRL"  # 环境名称

    has_continuous_action_space = True  # 是否为连续动作空间

    # 设置训练的参数
    max_ep_len = 500  # 每个episode最大步长
    max_training_timesteps = 200000  # 最大训练步长数

    print_freq = 150  # 每隔多少timesteps打印一次平均奖励
    log_freq = max_ep_len * 2  # 每隔多少timesteps记录一次日志
    save_model_freq = 3000  # 每隔多少timesteps保存一次模型

    action_std = 0.8  # 初始的动作分布的标准差（多元正态分布）
    action_std_decay_rate = 0.01  # 动作标准差衰减率（action_std -= action_std_decay_rate）
    min_action_std = 0.1  # 最小动作标准差（当标准差小于此值时停止衰减）
    action_std_decay_freq = 1000  # 衰减频率

    update_timestep = 600 * 1  # 每多少timesteps更新一次策略
    K_epochs = 10  # 每次更新PPO策略时优化K轮

    eps_clip = 0.25  # PPO中的裁剪参数
    gamma = 0.92  # 折扣因子（对于连续动作空间较高）

    lr_actor = 3e-4  # actor网络的学习率
    lr_critic = 3e-4  # critic网络的学习率

    random_seed = 47  # 如果需要随机种子，设置为非零值

    # 初始化环境，传入相关参数
    env = DRLEnvironment(viz_image_cv2=False, observation_type="god")

    # 获取状态空间和动作空间的维度
    state_dim = env.get_observation_space()[0]
    if has_continuous_action_space:
        action_dim = env.action_dim  # 对于连续空间，动作空间的维度
    else:
        action_dim = env.action_dim  # 对于离散空间，动作空间的维度

    # 设置日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取日志文件编号
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # 创建新的日志文件
    log_f_name = log_dir + '/PPO_A3' + env_name + "_log_" + str(run_num) + ".csv"

    run_num_pretrained = 1 # 修改此值以防止覆盖权重文件，仅仅用于作为保存模型的参数
    continue_training = False # 是否继续训练，若为True会加载之前的模型

    # 设置模型保存目录
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 设置模型的保存路径
    # checkpoint_path = directory + "ppo_bc_clone.pth"
    checkpoint_path = directory + "PPO_Action4_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("模型文件路径: " + checkpoint_path)

    # 如果需要，设置随机种子
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # 初始化PPO智能体
    ppo_agent = PPO_linear(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,action_std)

    # 记录训练开始时间
    start_time = datetime.now().replace(microsecond=0)

    # 创建并打开日志文件
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')  # 写入日志标题

    # 定义用于打印和记录的变量
    print_running_reward = 0
    print_running_episodes = 1
    log_running_reward = 0
    log_running_episodes = 1

    time_step = 0
    i_episode = 0

    # 如果继续训练，加载预训练模型
    if continue_training:
        print("加载模型: " + checkpoint_path)
        ppo_agent.load(checkpoint_path)

    env.init_race_environment()  # 初始化环境
    UPDATE_FLAG =False
    # 开始训练循环
    while time_step <= max_training_timesteps:

        state,_ = env.reset()  # 重置环境
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):
            # 使用PPO智能体选择动作
            action,action_logprob = ppo_agent.select_action(state)
            a_ = action.detach().cpu().numpy().flatten()
            a_[0] = a_[0]*7+7
            a_[1] = a_[1]*3
            a_[2] = a_[2]*8
            a_[3] = a_[3]*8
            next_state, reward, done,trunc,next_gate = env.step(a_)  # 环境根据动作返回新状态、奖励和是否结束
            next_gate = next_gate["next_gates"]
            ppo_agent.add_memory(state,action,action_logprob,reward,next_state,done)
            state = next_state

            time_step += 1
            current_ep_reward += reward

            if time_step % update_timestep == 0:
                UPDATE_FLAG = True
            # 如果是连续动作空间，则衰减动作标准差
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # 每隔一定时间步记录日志
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # 每隔一定时间步打印平均奖励
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode: {} \t Timestep: {} \t Reward: {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 1

            # 每隔一定时间步保存模型
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                print("模型已保存")
                print("训练时间: ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # 如果当前episode结束，则跳出循环
            if done:
                break
        # 每隔一定时间步更新PPO智能体
        if UPDATE_FLAG:
            print("---UPDATE----")
            ppo_agent.update(8)
            UPDATE_FLAG = False
        # ppo_agent.update(50)
        # 更新每个episode的奖励总和和次数
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1



def train_PPO():
    env_name = "DRL"  # 环境名称

    has_continuous_action_space = True  # 是否为连续动作空间

    # 设置训练的参数
    max_ep_len = 500  # 每个episode最大步长
    max_training_timesteps = 200000  # 最大训练步长数

    print_freq = 150  # 每隔多少timesteps打印一次平均奖励
    log_freq = max_ep_len * 2  # 每隔多少timesteps记录一次日志
    save_model_freq = 200  # 每隔多少timesteps保存一次模型

    action_std = 0.8  # 初始的动作分布的标准差（多元正态分布）
    action_std_decay_rate = 0.01  # 动作标准差衰减率（action_std -= action_std_decay_rate）
    min_action_std = 0.1  # 最小动作标准差（当标准差小于此值时停止衰减）
    action_std_decay_freq = 1000  # 衰减频率

    update_timestep = 300 * 1  # 每多少timesteps更新一次策略
    K_epochs = 30  # 每次更新PPO策略时优化K轮

    eps_clip = 0.25  # PPO中的裁剪参数
    gamma = 0.92  # 折扣因子（对于连续动作空间较高）

    lr_actor = 3e-4  # actor网络的学习率
    lr_critic = 3e-4  # critic网络的学习率

    random_seed = 47  # 如果需要随机种子，设置为非零值

    # 初始化环境，传入相关参数
    env = DRLEnvironment(viz_image_cv2=False, observation_type="god")

    # 获取状态空间和动作空间的维度
    state_dim = env.get_observation_space()[0]
    if has_continuous_action_space:
        action_dim = env.action_space[0]  # 对于连续空间，动作空间的维度
    else:
        action_dim = env.action_space[0]  # 对于离散空间，动作空间的维度

    # 设置日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取日志文件编号
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # 创建新的日志文件
    log_f_name = log_dir + '/PPO_Image' + env_name + "_log_" + str(run_num) + ".csv"

    run_num_pretrained = 1 # 修改此值以防止覆盖权重文件，仅仅用于作为保存模型的参数
    continue_training = True # 是否继续训练，若为True会加载之前的模型

    # 设置模型保存目录
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 设置模型的保存路径
    # checkpoint_path = directory + "ppo_bc_clone.pth"
    checkpoint_path = directory + "PPO_Image_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("模型文件路径: " + checkpoint_path)

    # 如果需要，设置随机种子
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    # 初始化PPO智能体
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,action_std)

    # 记录训练开始时间
    start_time = datetime.now().replace(microsecond=0)

    # 创建并打开日志文件
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')  # 写入日志标题

    # 定义用于打印和记录的变量
    print_running_reward = 0
    print_running_episodes = 1
    log_running_reward = 0
    log_running_episodes = 1

    time_step = 0
    i_episode = 0

    # 如果继续训练，加载预训练模型
    if continue_training:
        print("加载模型: " + checkpoint_path)
        ppo_agent.load(checkpoint_path)

    env.init_race_environment()  # 初始化环境

    # 开始训练循环
    while time_step <= max_training_timesteps:

        state = env.start_race()  # 开始新的比赛，获取初始状态
        current_ep_reward = 0
        for t in range(1, max_ep_len + 1):

            # 使用PPO智能体选择动作
            action,action_logprob = ppo_agent.select_action(state)
            a_ = action.detach().cpu().numpy().flatten()
            next_state, reward, done,next_gate = env.step(a_)  # 环境根据动作返回新状态、奖励和是否结束
            ppo_agent.add_memory(state,action,action_logprob,reward,next_state,done)
            state = next_state

            time_step += 1
            current_ep_reward += reward

            # 每隔一定时间步更新PPO智能体
            if time_step % update_timestep == 0:
                ppo_agent.update(20)

            # 如果是连续动作空间，则衰减动作标准差
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            # 每隔一定时间步记录日志
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0

            # 每隔一定时间步打印平均奖励
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode: {} \t Timestep: {} \t Reward: {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 1

            # 每隔一定时间步保存模型
            if time_step % save_model_freq == 0:
                ppo_agent.save(checkpoint_path)
                print("模型已保存")
                print("训练时间: ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # 如果当前episode结束，则跳出循环
            if done:
                break
        # ppo_agent.update(50)
        # 更新每个episode的奖励总和和次数
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

        env.reset()  # 重置环境
        time.sleep(1)  # 等待3秒后开始下一个episode
def train_SAC():
    env_name = "DRL"  # 环境名称

    has_continuous_action_space = True  # 是否为连续动作空间

    # 设置训练的参数
    max_ep_len = 300  # 每个episode最大步长
    max_training_timesteps = 20000  # 最大训练步长数

    print_freq = 50 # 每隔多少timesteps打印一次平均奖励
    log_freq = 50 * 2  # 每隔多少timesteps记录一次日志
    save_model_freq = 300  # 每隔多少timesteps保存一次模型


    update_timestep = 5  # 每多少timesteps更新一次策略

    random_seed = 47  # 如果需要随机种子，设置为非零值

    # 初始化环境，传入相关参数
    env = DRLEnvironment(viz_image_cv2=False, observation_type="god")

    # 获取状态空间和动作空间的维度
    state_dim = env.get_observation_space()[0]
    if has_continuous_action_space:
        action_dim = env.action_space[0]  # 对于连续空间，动作空间的维度
    else:
        action_dim = env.action_space[0]  # 对于离散空间，动作空间的维度

    # 设置日志目录
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # 获取日志文件编号
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    # 创建新的日志文件
    log_f_name = log_dir + '/SAC_' + env_name + "_log_" + str(run_num) + ".csv"

    run_num_pretrained = 1 # 修改此值以防止覆盖权重文件，仅仅用于作为保存模型的参数
    continue_training = True # 是否继续训练，若为True会加载之前的模型

    # 设置模型保存目录
    directory = "models"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # 设置模型的保存路径
    checkpoint_path = directory + "SAC_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    # load_path = directory + "SAC_base.pth"
    load_path = directory + "SAC_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("模型文件路径: " + checkpoint_path)

    # 如果需要，设置随机种子
    if random_seed:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)

    agent = SAC(action_dim,state_dim,2)

    # 记录训练开始时间
    start_time = datetime.now().replace(microsecond=0)

    # 创建并打开日志文件
    log_f = open(log_f_name, "w+")
    log_f.write('episode,timestep,reward\n')  # 写入日志标题

    # 定义用于打印和记录的变量
    print_running_reward = 0
    print_running_episodes = 1
    log_running_reward = 0
    log_running_episodes = 1

    time_step = 0
    i_episode = 0

    best_gate = 1
    # 如果继续训练，加载预训练模型
    if continue_training:
        print("加载模型: " + load_path)
        agent.load(load_path)

    env.init_race_environment()  # 初始化环境

    # 开始训练循环
    while time_step <= max_training_timesteps:
        # if time_step % update_timestep == 0:
        state = env.start_race()  # 开始新的比赛，获取初始状态
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
            action = agent.select_action(state)
            next_state, reward, done,next_gate = env.step(action)  # 环境根据动作返回新状态、奖励和是否结束
            best_gate = max(best_gate, next_gate+1)

            # 保存奖励和终止信息
            agent.append_sample(next_gate,state, action, reward, next_state, done) #1
            state = next_state
            #
            # with open("traininglog.txt","a") as f:
            #     f.write(f"state:{state},action:{action},reward:{reward}\n")

            time_step += 1
            current_ep_reward += reward
            # 每隔一定时间步记录日志
            if time_step % log_freq == 0:
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                log_running_reward = 0
                log_running_episodes = 1

            if time_step % 50 == 0:
                agent.update()

            # 每隔一定时间步打印平均奖励
            if time_step % print_freq == 0:
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
                print("Episode: {} \t Timestep: {} \t Reward: {}".format(i_episode, time_step, print_avg_reward))
                print_running_reward = 0
                print_running_episodes = 1

            # 每隔一定时间步保存模型
            if time_step % save_model_freq == 0:
                agent.save(checkpoint_path)
                print("模型已保存")
                print("训练时间: ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # 如果当前episode结束，则跳出循环
            if done:
                break

        # 更新每个episode的奖励总和和次数
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        agent.turn_offline_sample()
        # for i in range(best_gate):
        #     agent.mini_update(i)
        env.reset()  # 重置环境
        time.sleep(3)  # 等待3秒后开始下一个episode

    log_f.close()  # 关闭日志文件

def test():
    # 设置环境和测试参数
    env_name = "DRL"
    has_continuous_action_space = True  # 是否为连续动作空间
    max_ep_len = 300  # 每个episode最大步数
    action_std = 0.10  # 动作标准差，使用训练时的相同标准差

    total_test_episodes = 10  # 测试时的总episode数

    # PPO算法的超参数（不再训练，只进行测试）
    K_epochs = 80  # 每个更新的epoch数
    eps_clip = 0.2  # PPO中的裁剪参数
    gamma = 0.99  # 折扣因子

    lr_actor = 0.0001  # actor网络的学习率
    lr_critic = 0.0001  # critic网络的学习率

    # 初始化环境
    env = DRLEnvironment(viz_image_cv2=False, observation_type="lidar")

    # 获取状态空间和动作空间的维度
    state_dim = env.get_observation_space()[0]
    if has_continuous_action_space:
        action_dim = env.action_space[0]  # 对于连续空间，动作空间的维度
    else:
        action_dim = env.action_space[0]  # 对于离散空间，动作空间的维度

    # 初始化PPO智能体
    ppo_agent = PPO_linear(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space,
                    action_std)

    # 设置预训练模型的路径
    random_seed = 47  # 设置随机种子，加载特定的checkpoint
    run_num_pretrained = 5  # 设置预训练模型编号

    directory = "models" + '/' + env_name + '/'  # 预训练模型所在的目录
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("加载预训练模型: " + checkpoint_path)

    # 加载预训练模型
    ppo_agent.load(checkpoint_path)
    print("--------------------------------------------------------------------------------------------")

    # 记录测试时的奖励
    test_running_reward = 0

    # 初始化环境
    env.init_race_environment()

    # 测试循环
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0  # 每个episode的总奖励
        state = env.start_race()  # 获取初始状态

        # 进行每个episode的动作选择和环境交互
        for t in range(1, max_ep_len + 1):
            action,_ = ppo_agent.select_action(state)  # 根据当前状态选择动作
            state, reward, done = env.step(action)  # 执行动作并获取新的状态、奖励和是否结束标志
            ep_reward += reward  # 累积奖励

            if done:  # 如果当前episode结束，跳出循环
                break

        # 清空PPO智能体的buffer（虽然在测试过程中不需要buffer，但最好清理）
        ppo_agent.buffer.clear()

        # 累加当前episode的奖励
        test_running_reward += ep_reward

        # 打印当前episode的奖励
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))

        # 重置环境并等待3秒后开始下一个episode
        env.reset()
        time.sleep(3)

    print("============================================================================================")

    # 计算并打印平均奖励
    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("平均奖励: " + str(avg_test_reward))

    print("============================================================================================")


def plot():
    env_name = 'DRL'  # 环境名称

    fig_num = 0  # 图表编号，避免覆盖图形
    plot_avg = False  # 是否绘制所有运行的平均值曲线，否则绘制每次运行的单独曲线
    fig_width = 10  # 图表宽度
    fig_height = 6  # 图表高度

    # 平滑处理奖励数据的窗口大小和参数
    window_len_smooth = 20
    min_window_len_smooth = 1
    linewidth_smooth = 1.5
    alpha_smooth = 1

    window_len_var = 5
    min_window_len_var = 1
    linewidth_var = 2
    alpha_var = 0.1

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'olive', 'brown', 'magenta', 'cyan', 'crimson','gray', 'black']

    # 创建保存图形的目录
    figures_dir = "plots"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # 环境名称对应的子目录
    figures_dir = figures_dir + '/' + env_name + '/'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig_save_path = figures_dir + '/PPO_' + env_name + '_fig_' + str(fig_num) + '.png'

    # 获取日志文件的数量
    log_dir = "logs_to_plot" + '/' + env_name + '/'
    current_num_files = next(os.walk(log_dir))[2]
    num_runs = len(current_num_files)

    all_runs = []

    # 读取所有的日志文件
    for run_num in range(num_runs):
        log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"
        data = pd.read_csv(log_f_name)  # 读取csv文件
        data = pd.DataFrame(data)
        all_runs.append(data)  # 将每次运行的奖励数据存入all_runs列表

    ax = plt.gca()  # 获取当前的坐标轴对象

    if plot_avg:
        # 如果plot_avg为True，计算所有运行的平均奖励并绘制
        df_concat = pd.concat(all_runs)  # 合并所有的运行数据
        df_concat_groupby = df_concat.groupby(df_concat.index)  # 按索引分组
        data_avg = df_concat_groupby.mean()  # 计算平均值

        # 对奖励进行平滑处理
        data_avg['reward_smooth'] = data_avg['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
        data_avg['reward_var'] = data_avg['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

        # 绘制平均奖励曲线和方差曲线
        data_avg.plot(kind='line', x='timestep', y='reward_smooth', ax=ax, color=colors[0], linewidth=linewidth_smooth, alpha=alpha_smooth)
        data_avg.plot(kind='line', x='timestep', y='reward_var', ax=ax, color=colors[0], linewidth=linewidth_var, alpha=alpha_var)

        # 修改图例，确保只显示平滑奖励曲线
        handles, labels = ax.get_legend_handles_labels()
        ax.legend([handles[0]], ["reward_avg_" + str(len(all_runs)) + "_runs"], loc=2).remove()

    else:
        # 如果plot_avg为False，绘制每次训练运行的奖励曲线
        for i, run in enumerate(all_runs):
            # 平滑处理每个运行的奖励数据
            run['reward_smooth_' + str(i)] = run['reward'].rolling(window=window_len_smooth, win_type='triang', min_periods=min_window_len_smooth).mean()
            run['reward_var_' + str(i)] = run['reward'].rolling(window=window_len_var, win_type='triang', min_periods=min_window_len_var).mean()

            # 绘制平滑后的奖励曲线和方差曲线
            run.plot(kind='line', x='timestep', y='reward_smooth_' + str(i), ax=ax, color=colors[i % len(colors)], linewidth=linewidth_smooth, alpha=alpha_smooth)
            run.plot(kind='line', x='timestep', y='reward_var_' + str(i), ax=ax, color=colors[i % len(colors)], linewidth=linewidth_var, alpha=alpha_var)

        # 仅将平滑奖励曲线添加到图例
        handles, labels = ax.get_legend_handles_labels()
        new_handles = []
        new_labels = []
        for i in range(len(handles)):
            if(i % 2 == 0):  # 只保留每隔一个的曲线（平滑奖励曲线）
                new_handles.append(handles[i])
                new_labels.append(labels[i])
        ax.legend(new_handles, new_labels, loc=2).remove()

    # 设置坐标轴网格、标签和标题
    ax.grid(color='gray', linestyle='-', linewidth=1, alpha=0.2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Rewards", fontsize=12)
    plt.title(env_name, fontsize=14)

    # 设置图形尺寸并保存
    fig = plt.gcf()
    fig.set_size_inches(fig_width, fig_height)
    plt.savefig(fig_save_path)

    # 显示图形
    plt.show()


def main(args):
    if args.mode == 'train':
        train()
         
    if args.mode == 'test':
        test()
        
    if args.mode == 'plot':
        plot()

    if args.mode =="train_SAC":
        train_SAC()
    if args.mode =="train_PPO":
        train_PPO()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        choices=[
            "test",
            "plot"
        ],
        default="train",
    )
    
    args = parser.parse_args()
    main(args)