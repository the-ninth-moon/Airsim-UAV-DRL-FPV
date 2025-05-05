import numpy as np
import  gymnasium as gym
from gymnasium import Env, spaces
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def _arrow3d(ax, xs, ys, zs, *args, **kwargs):
    '''Draw an arrow with matplotlib.patches.FancyArrowPatch in 3d.'''
    arrow = Arrow3D(xs, ys, zs, *args, **kwargs)
    ax.add_artist(arrow)


class RelativePosEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30} # 添加支持的渲染模式和帧率

    def __init__(self, render_mode=None):
        super().__init__()
        # 状态空间定义
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(67,), dtype=np.float32
        )

        # 动作空间定义
        self.action_space = spaces.Box(
            low=np.array([-5,-3,-5]),
            high=np.array([5,3,5]),
            dtype=np.float64
        ) #连续

        # 环境参数
        self.max_steps = 200  # 每个episode最大步数
        self.current_step = 0
        self.target_pos = None
        self.current_pos = None
        self.current_relative_pos = None
        self.distance_threshold = 0.5  # 距离目标点的阈值

        self.render_mode = render_mode
        self.fig = None
        self.ax = None


    def reset(self, seed=None, options=None, target_pos=None, current_pos=None):
        super().reset(seed=seed)
        self.current_step = 0

        # 设置目标地点和当前地点
        if target_pos is None:
            self.target_pos = self.np_random.uniform(-20, 20, size=3)  # 随机目标点
        else:
            self.target_pos = np.array(target_pos)

        if current_pos is None:
            self.current_pos = self.np_random.uniform(-20, 20, size=3) # 随机起始点
        else:
            self.current_pos = np.array(current_pos)

        self.current_relative_pos = self.target_pos - self.current_pos

        # 生成随机初始状态 (保持其他维度随机)
        img_state = self.np_random.normal(size=4)  # 4维图像特征
        orientation = self._random_quaternion()  # 4维姿态四元数
        gate_orientation = self._random_quaternion()  # 4维门框姿态

        # 构建前15维状态，现在 relative_pos 是基于设定的 target_pos 和 current_pos
        state_15 = np.concatenate([
            # img_state,
            self.current_relative_pos,
            # orientation,
            # gate_orientation
        ])

        # 剩余64维随机填充
        state_64 = self.np_random.normal(size=64)

        observation = np.concatenate([state_15, state_64]).astype(np.float32)

        if self.render_mode == "human":
            self._render_frame()

        return observation, {}

    def step(self, action):
        self.current_step += 1

        # 根据动作更新当前位置
        self.current_pos += action

        # 重新计算相对位置
        self.current_relative_pos = self.target_pos - self.current_pos

        # 计算奖励
        reward = self._calculate_reward(action)

        # 生成新状态（保持 relative_pos 和 current_pos 相关，其他随机更新）
        new_state = self._generate_new_state()

        # 终止条件判断: 到达目标点附近或超过最大步数
        terminated = self._check_if_reached_target()
        truncated = self.current_step >= self.max_steps

        info = {"is_success": terminated} # For tracking success in training
        # print(f"pos:{np.round(self.current_pos, 2)}, target:{np.round(self.target_pos, 2)}, action:{np.round(action, 2)}, reward:{reward:.2f}, terminal:{terminated}, truncated:{truncated}")

        ta = self.current_relative_pos/ (np.linalg.norm(self.current_relative_pos) + 0.0001)
        tb = action/(np.linalg.norm(action) + 0.0001)
        similar = (np.dot(ta,tb) / (np.linalg.norm(ta)*np.linalg.norm(tb) + 1e-4))
        reward += np.linalg.norm(action)/20-0.4
        # print(f"relative_pos:{self.current_relative_pos}, action:{np.round(action, 2)},similar:{similar}, reward:{reward:.2f}, terminal:{terminated}, truncated:{truncated}")

        observation = new_state

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, action):
        """奖励函数：到达目标附近给高奖励，否则基于动作与相对位置的相似度"""
        if self._check_if_reached_target():
            return 10.0  # 到达目标给高奖励
        else:
            return self._calculate_similarity_reward(action)

    def _calculate_similarity_reward(self, action):
        """核心奖励计算逻辑：正则化相似度"""
        a = np.array(action)
        r = self.current_relative_pos

        # 正则化处理
        a_norm = np.linalg.norm(a)
        r_norm = np.linalg.norm(r)

        if a_norm < 1e-6 or r_norm < 1e-6:
            return -0.1  # 避免除零，并略微惩罚静止不动? 也可以返回 0

        a_normalized = a / a_norm
        r_normalized = r / r_norm

        # 余弦相似度奖励（方向一致性）， 范围 [-1, 1]
        cosine_similarity = np.dot(a_normalized, r_normalized)

        return cosine_similarity # 奖励范围调整到 [-1, 1]

    def _check_if_reached_target(self):
        """检查是否到达目标点附近"""
        distance = np.linalg.norm(self.current_relative_pos)
        return distance < self.distance_threshold

    def _random_quaternion(self):
        """生成随机单位四元数"""
        q = self.np_random.normal(size=4)
        return q / np.linalg.norm(q)

    def _generate_new_state(self):
        """生成新状态（更新除 relative_pos 相关的状态维度）"""
        img_state = self.np_random.normal(size=4)
        orientation = self._random_quaternion()
        gate_orientation = self._random_quaternion()

        state_15 = np.concatenate([
            # img_state,
            self.current_relative_pos, # 保持更新后的 relative_pos
            # orientation,
            # gate_orientation
        ])

        state_64 = self.np_random.normal(size=64)
        return np.concatenate([state_15, state_64]).astype(np.float32)

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render without specifying render_mode. "
                "Your environment will run with no rendering. "
            )
        else:
            return self._render_frame()

    def _render_frame(self):
        if self.fig is None:
            self.fig = plt.figure(figsize=(8, 8))
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlabel('X')
            self.ax.set_ylabel('Y')
            self.ax.set_zlabel('Z')
            self.ax.set_xlim([-30, 30])
            self.ax.set_ylim([-30, 30])
            self.ax.set_zlim([-30, 30])
            self.ax.set_aspect('equal') # 保证三轴比例一致

        self.ax.clear() # 清空之前的图像

        # 绘制目标点 (红色)
        self.ax.scatter(*self.target_pos, color='red', s=100, label='Target')
        # 绘制当前位置 (蓝色)
        self.ax.scatter(*self.current_pos, color='blue', s=100, label='Current Position')

        # 绘制从当前位置到目标位置的箭头
        _arrow3d(self.ax,
                 [self.current_pos[0], self.target_pos[0]],
                 [self.current_pos[1], self.target_pos[1]],
                 [self.current_pos[2], self.target_pos[2]],
                 mutation_scale=20, lw=1, arrowstyle="-|>", color="black", label='Relative Vector')


        self.ax.legend() # 显示图例
        self.fig.canvas.draw()

        if self.render_mode == "human":
            plt.pause(0.01) # 暂停一小段时间，用于动画效果
            return None
        elif self.render_mode == "rgb_array":
            return np.array(self.fig.canvas.get_renderer()._renderer)
        else:
            super().render()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == '__main__':
    env = RelativePosEnv(render_mode="human") # 初始化环境时指定 render_mode="human"
    obs, _ = env.reset()
    for _ in range(200): # 运行几个episode
        action = env.action_space.sample() # 随机动作
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            env.reset()
    env.close()