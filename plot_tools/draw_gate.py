import numpy as np
import re # 导入正则表达式库
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation # 需要 scipy 库来处理四元数
import matplotlib.animation as animation
def parse_gate_info(filepath="gate_info.txt"):
    gates_data = []
    current_gate = {}
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line: # 跳过空行
                continue

            # 检查是否是门框名称行 (例如 "Gate0:")
            gate_match = re.match(r"^(Gate\d+):$", line)
            if gate_match:
                # 如果当前正在处理一个门框，并且遇到了新的门框名称，
                # 先保存上一个门框的信息 (如果有效)
                if 'name' in current_gate and 'position' in current_gate and 'orientation' in current_gate:
                    gates_data.append(current_gate)
                # 开始处理新的门框
                current_gate = {'name': gate_match.group(1)}
                continue # 继续下一行

            # 检查是否是位置信息行
            # 使用更鲁棒的正则表达式匹配浮点数（包括负数和科学计数法）
            pos_match = re.match(
                r"position_x_val:(-?[\d\.eE+-]+)\s+"
                r"position_y_val:(-?[\d\.eE+-]+)\s+"
                r"position_z_val:(-?[\d\.eE+-]+)", line
            )
            if pos_match and 'name' in current_gate:
                try:
                    pos_x = float(pos_match.group(1))
                    pos_y = float(pos_match.group(2))
                    pos_z = float(pos_match.group(3))
                    current_gate['position'] = np.array([pos_x, pos_y, pos_z])
                except ValueError:
                    print(f"警告: 无法解析门 '{current_gate.get('name', '未知')}' 的位置数据行: {line}")
                continue # 继续下一行

            # 检查是否是姿态信息行
            orient_match = re.match(
                r"orientation_x_val:(-?[\d\.eE+-]+)\s+"
                r"orientation_y_val:(-?[\d\.eE+-]+)\s+"
                r"orientation_z_val:(-?[\d\.eE+-]+)\s+"
                r"orientation_w_val:(-?[\d\.eE+-]+)", line
            )
            if orient_match and 'name' in current_gate:
                try:
                    orient_x = float(orient_match.group(1))
                    orient_y = float(orient_match.group(2))
                    orient_z = float(orient_match.group(3))
                    orient_w = float(orient_match.group(4))
                    # 注意：scipy.spatial.transform.Rotation 需要 [x, y, z, w] 顺序
                    current_gate['orientation'] = np.array([orient_x, orient_y, orient_z, orient_w])
                except ValueError:
                    print(f"警告: 无法解析门 '{current_gate.get('name', '未知')}' 的姿态数据行: {line}")
                continue # 继续下一行

        # 添加文件末尾的最后一个门框信息
        if 'name' in current_gate and 'position' in current_gate and 'orientation' in current_gate:
            gates_data.append(current_gate)

    except FileNotFoundError:
        print(f"错误: 文件 '{filepath}' 未找到。")
        return []
    except Exception as e:
        print(f"解析文件时发生错误: {e}")
        return []

    print(f"成功从 '{filepath}' 解析了 {len(gates_data)} 个门框信息。")
    return gates_data

def plot_gate_matplotlib(ax, gate_pos, gate_quat, name, length=5, width=5.5, thickness=0.2, color='r'):
    """
    使用 Matplotlib 在 3D 坐标轴上绘制一个门框。

    Args:
        ax (Axes3D): Matplotlib 的 3D 坐标轴对象。
        gate_pos (np.array): 门框中心的位置 [x, y, z]。
        gate_quat (np.array): 门框的姿态四元数 [x, y, z, w]。
        name (str): 门框的名称，用于标注。
        length (float): 门框沿本地 Y 轴的长度（通常是高的那个方向）。
        width (float): 门框沿本地 Z 轴的宽度（通常是宽的那个方向）。
        thickness (float): 门框沿本地 X 轴的厚度。
        color (str): 门框线条的颜色。
    """
    # 从四元数创建旋转对象 (scipy 使用 [x, y, z, w] 顺序)
    # 需要处理单位四元数可能存在的微小误差
    norm = np.linalg.norm(gate_quat)
    if norm < 1e-6:
        print(f"警告: 门 '{name}' 的四元数接近零，无法归一化。跳过绘制。")
        return
    rotation = Rotation.from_quat(gate_quat / norm)

    # 定义门框的8个顶点（在门框局部坐标系中）
    # thickness 沿 X 轴, length 沿 Y 轴, width 沿 Z 轴
    half_l, half_w, half_t = length / 2, width / 2, thickness / 2
    vertices_local = np.array([
        [-half_t,  half_l,  half_w], # 0: 后左上
        [-half_t,  half_l, -half_w], # 1: 后左下
        [-half_t, -half_l,  half_w], # 2: 后右上
        [-half_t, -half_l, -half_w], # 3: 后右下
        [ half_t,  half_l,  half_w], # 4: 前左上
        [ half_t,  half_l, -half_w], # 5: 前左下
        [ half_t, -half_l,  half_w], # 6: 前右上
        [ half_t, -half_l, -half_w]  # 7: 前右下
    ])

    # 将局部顶点坐标旋转并平移到世界坐标系
    vertices_world = rotation.apply(vertices_local) + gate_pos

    # 定义构成门框框架的12条边（连接顶点的索引）
    edges = [
        [0, 1], [0, 2], [1, 3], [2, 3], # 后框
        [4, 5], [4, 6], [5, 7], [6, 7], # 前框
        [0, 4], [1, 5], [2, 6], [3, 7]  # 连接前后框
    ]

    # 绘制门框的边
    for edge in edges:
        points = vertices_world[edge] # 获取边的两个端点
        ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color)

    # 在门框中心稍微偏移的位置添加名称标注
    # 简单的偏移量，可能需要根据视角调整
    text_pos = gate_pos + rotation.apply([thickness, 0, 0]) # 稍微向前偏移
    ax.text(text_pos[0], text_pos[1], text_pos[2], name, color=color)

def plot_drone_trajectory(ax, trajectory_points, color='b', marker='.', linestyle='-'):
    if trajectory_points is not None and len(trajectory_points) > 0:
        ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
                color=color, marker=marker, linestyle=linestyle, label="Drone Trajectory")
        ax.scatter(trajectory_points[0, 0], trajectory_points[0, 1], trajectory_points[0, 2],
                   color='lime', s=100, marker='o', label="Start") # 标记起点
        ax.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], trajectory_points[-1, 2],
                   color='magenta', s=100, marker='X', label="End")   # 标记终点
def update_line(num, trajectory_points, line, point_marker):
    """
    FuncAnimation 的更新函数。

    Args:
        num (int): 当前帧编号 (从 0 开始)。
        trajectory_points (np.array): N x 3 的轨迹点数组。
        line (matplotlib.lines.Line3D): 要更新的轨迹线对象。
        point_marker (matplotlib.collections.PathCollection): 要更新的当前点标记对象。
    """
    # 更新轨迹线，显示从起点到当前点的所有线段
    line.set_data(trajectory_points[:num+1, 0], trajectory_points[:num+1, 1])
    line.set_3d_properties(trajectory_points[:num+1, 2])

    # 更新当前点标记的位置
    # 使用 _offsets3d 来设置 3D 散点的位置 (注意是列表套列表或元组套元组)
    point_marker._offsets3d = ([trajectory_points[num, 0]], [trajectory_points[num, 1]], [trajectory_points[num, 2]])

    # 返回已修改的艺术家对象列表 (对于 blitting 优化)
    return line, point_marker

def plot_trajectory_animation(fig, ax, trajectory_points, interval=1, color='b', marker_color='red', **plot_kwargs):
    """
    在 3D 坐标轴上动态绘制无人机轨迹。

    Args:
        fig (matplotlib.figure.Figure): Matplotlib 图形对象。
        ax (Axes3D): Matplotlib 的 3D 坐标轴对象。
        trajectory_points (np.array): N x 3 的轨迹点数组。
        interval (int): 动画帧之间的延迟（毫秒）。
        color (str): 轨迹线的颜色。
        marker_color (str): 当前移动点的颜色。
        **plot_kwargs: 其他传递给 ax.plot 的参数 (如 linewidth)。
    """
    if trajectory_points is None or trajectory_points.shape[0] < 2:
        print("轨迹点不足，无法创建动画。")
        # 仍然可以绘制单个点（如果存在）
        if trajectory_points is not None and trajectory_points.shape[0] == 1:
             ax.scatter(trajectory_points[0, 0], trajectory_points[0, 1], trajectory_points[0, 2],
                       color=marker_color, s=100, marker='o', label="Single Point")
        return None # 返回 None 表示没有创建动画对象

    # 初始化轨迹线 (开始时只包含第一个点，或者为空)
    # 使用 plot 返回的列表中的第一个元素
    line, = ax.plot([trajectory_points[0, 0]], [trajectory_points[0, 1]], [trajectory_points[0, 2]],
                   color=color, label="Drone Trajectory", **plot_kwargs)

    # 初始化当前点标记
    point_marker = ax.scatter([trajectory_points[0, 0]], [trajectory_points[0, 1]], [trajectory_points[0, 2]],
                              color=marker_color, s=80, marker='o', label="Current Position", zorder=10) # zorder 确保在轨迹线上方

    # 标记固定的起点和终点（可选，如果希望动画结束后保留标记）
    ax.scatter(trajectory_points[0, 0], trajectory_points[0, 1], trajectory_points[0, 2],
               color='lime', s=100, marker='o', label="Start", alpha=0.8, zorder=5)
    ax.scatter(trajectory_points[-1, 0], trajectory_points[-1, 1], trajectory_points[-1, 2],
               color='magenta', s=100, marker='X', label="End", alpha=0.8, zorder=5)

    # 创建动画对象
    # frames=len(trajectory_points) 表示动画总共有多少帧
    # fargs 传递额外参数给 update_line 函数
    # interval 控制动画速度
    # blit=True 尝试优化绘图，只重绘变化的部分 (有时可能引起问题)
    # repeat=False 让动画只播放一次
    print(len(trajectory_points))
    ani = animation.FuncAnimation(fig, update_line, frames=len(trajectory_points),
                                  fargs=(trajectory_points, line, point_marker),
                                  interval=interval, blit=True, repeat=False)

    return ani # 返回动画对象，需要保持引用以防被垃圾回收


def parse_trajectory_info(filepath="trajectory_log.txt"):
    """
    从指定的文本文件中读取无人机轨迹点。

    文件格式应为每行包含:
    position_x:VALUE position_y:VALUE position_z:VALUE
    其中 VALUE 是浮点数。

    Args:
        filepath (str): 轨迹记录文件的路径。

    Returns:
        numpy.ndarray: 一个 N x 3 的 NumPy 数组，其中 N 是轨迹点的数量。
                       每一行代表一个 [x, y, z] 坐标。
                       如果文件未找到或无法解析任何有效行，则返回一个空的 (0, 3) NumPy 数组。
    """
    trajectory_points = []
    # 正则表达式匹配 x, y, z 坐标值 (允许负数和科学计数法)
    # \s+ 匹配一个或多个空格
    line_pattern = re.compile(
        r"position_x:(-?[\d\.eE+-]+)\s+"
        r"position_y:(-?[\d\.eE+-]+)\s+"
        r"position_z:(-?[\d\.eE+-]+)"
    )

    try:
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                line = line.strip() # 去除前后空白字符
                if not line: # 跳过空行
                    continue

                match = line_pattern.match(line)
                if match:
                    try:
                        # 提取匹配到的组并转换为浮点数
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        trajectory_points.append([x, y, z])
                    except ValueError:
                        print(f"警告: 无法将文件 '{filepath}' 第 {i+1} 行的坐标值转换为浮点数: {line}")
                    except IndexError:
                         print(f"警告: 文件 '{filepath}' 第 {i+1} 行的正则表达式匹配组不足: {line}") # 一般不会发生，除非 pattern 写错
                else:
                    print(f"警告: 文件 '{filepath}' 第 {i+1} 行格式不匹配: {line}")

    except FileNotFoundError:
        print(f"错误: 轨迹文件 '{filepath}' 未找到。")
        # 返回一个空的 (0, 3) 数组
        return np.empty((0, 3))
    except Exception as e:
        print(f"读取或解析轨迹文件 '{filepath}' 时发生错误: {e}")
        # 返回一个空的 (0, 3) 数组
        return np.empty((0, 3))

    if not trajectory_points:
        print(f"警告: 未能从文件 '{filepath}' 中解析出任何有效的轨迹点。")
        # 返回一个空的 (0, 3) 数组
        return np.empty((0, 3))
    else:
        print(f"成功从 '{filepath}' 解析了 {len(trajectory_points)} 个轨迹点。")
        # 将列表转换为 NumPy 数组并返回
        return np.array(trajectory_points)

# --- 主程序 ---
if __name__ == "__main__":
    # 1. 解析门框信息
    gate_file = "gate_info_s.txt"
    # --- 如果 gate_info.txt 不存在，先创建它 ---
    gates = parse_gate_info(gate_file)

    show_animation = True

    if not gates:
        print("未能加载门框信息，无法绘图。")
    else:
        # 2. 创建 Matplotlib 3D 图形和坐标轴
        fig = plt.figure(figsize=(10, 8)) # 设置图形大小
        ax = fig.add_subplot(111, projection='3d')

        all_vertices = [] # 用于计算坐标轴范围

        # 3. 绘制所有门框
        for i, gate in enumerate(gates):
            plot_gate_matplotlib(ax, gate['position'], gate['orientation'], gate['name'], color=plt.cm.viridis(i / len(gates))) # 使用不同颜色区分
            # 在计算边界时也需要考虑顶点
            rotation = Rotation.from_quat(gate['orientation'] / np.linalg.norm(gate['orientation']))
            half_l, half_w, half_t = 5 / 2, 5.5 / 2, 0.2 / 2 # 使用 plot_gate_matplotlib 中的默认值或实际值
            vertices_local = np.array([
                [-half_t,  half_l,  half_w], [-half_t,  half_l, -half_w],
                [-half_t, -half_l,  half_w], [-half_t, -half_l, -half_w],
                [ half_t,  half_l,  half_w], [ half_t,  half_l, -half_w],
                [ half_t, -half_l,  half_w], [ half_t, -half_l, -half_w]
            ])
            vertices_world = rotation.apply(vertices_local) + gate['position']
            all_vertices.extend(vertices_world)


        # --- (可选) 添加无人机轨迹数据 ---
        # 这是一个示例轨迹数据，你需要用你实际的数据替换它
        trajectory = parse_trajectory_info("trac_s.txt")  # 使用你的文件名
        if trajectory is not None and not show_animation:
            plot_drone_trajectory(ax, trajectory, color='blue', marker='.')
        # 4. 设置坐标轴标签和标题
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("AirSim Gates Visualization")
        # 5. 自动调整坐标轴范围以包含所有门框和轨迹
        if all_vertices:
            all_points = np.array(all_vertices)
            max_range = np.array([all_points[:,0].max()-all_points[:,0].min(),
                                  all_points[:,1].max()-all_points[:,1].min(),
                                  all_points[:,2].max()-all_points[:,2].min()]).max() / 2.0

            mid_x = (all_points[:,0].max()+all_points[:,0].min()) * 0.5
            mid_y = (all_points[:,1].max()+all_points[:,1].min()) * 0.5
            mid_z = (all_points[:,2].max()+all_points[:,2].min()) * 0.5
            ax.set_xlim(mid_x - max_range, mid_x + max_range)
            ax.set_ylim(mid_y - max_range, mid_y + max_range)
            ax.set_zlim(mid_z - max_range, mid_z + max_range)
        else:
             # 如果没有点，设置一个默认范围
             ax.set_xlim(-10, 10)
             ax.set_ylim(-10, 10)
             ax.set_zlim(-10, 0)


        # 设置 Z 轴反向（通常 AirSim Z 轴向下为正，但 Matplotlib 向上为正）
        ax.invert_zaxis() # 取消注释以反转 Z 轴
        ax.invert_yaxis() # 取消注释以反转 Z 轴
        # 添加图例（如果绘制了轨迹）
        if trajectory is not None and not show_animation:
            ax.legend()
        else:
            drone_animation = plot_trajectory_animation(fig, ax, trajectory, interval=30, color='deepskyblue',
                                                        marker_color='orangered', linewidth=2)
            drone_animation.save("2.gif")
        # 添加图例 (包含动画和静态元素)
        ax.legend()
        # 显示图形
        plt.show()
        # 设置视角（可选）
        # ax.view_init(elev=20., azim=-35)
        # 显示图形 (窗口可交互，用鼠标旋转)
