import airsimdroneracinglab as airsim
import numpy as np
import cv2
import threading
import time
import random
import math
# import open3d as o3d
import os


def detect_largest_door_in_image(image):
    """
    检测图像中的最大矩形（可能代表门）并返回其位置信息。
    输入:
        image: 输入图像，形状为(240, 320, 3)
    输出:
        detected: 如果检测到门，返回True；否则返回False。
        largest_door_info: 最大矩形的信息，包括顶点、角度、位置等。
    """
    # 将图像转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 应用高斯模糊，减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 使用Canny边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 找到图像中的轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_door_info = None  # 用于保存最大矩形的信息
    max_area = 0  # 当前最大面积

    # 遍历所有轮廓
    for contour in contours:
        # 计算轮廓的面积，较小的区域忽略
        area = cv2.contourArea(contour)
        if area < 200:  # 根据实际情况调整这个面积限制
            continue

        # 通过最小外接矩形获取轮廓的矩阵信息
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)  # 获取矩形的四个顶点
        box = np.int32(box)  # 将顶点转为整数（修复了np.int0问题）

        # 检查矩形的长宽比，宽度和高度没有限制，可以为任何值
        width, height = rect[1]

        # 如果当前矩形的面积大于已知的最大面积，则更新最大矩形
        if area > max_area:
            max_area = area
            largest_door_info = {
                'vertices': box,  # 矩形的四个顶点
                'angle': rect[2],  # 矩形的角度
                'position': rect[0],  # 矩形中心点
                'width': width,  # 矩形宽度
                'height': height,  # 矩形高度
            }
    image_copy = image.copy()  # 创建图像副本
    if largest_door_info is None:
        return False, None, image_copy  # 如果没有检测到任何矩形

    # 在图像副本上画出最大矩形（门）的检测结果

    cv2.drawContours(image_copy, [largest_door_info['vertices']], 0, (0, 255, 0), 3)
    # 画矩形中心点
    cv2.circle(image_copy, (int(largest_door_info['position'][0]), int(largest_door_info['position'][1])), 5,
               (0, 0, 255), -1)

    return True, largest_door_info, image_copy


def point_to_line_distance(P, A, B):
    # 向量 AB 和 AP
    AB = B - A
    AP = P - A

    # 计算叉积
    cross_product = np.cross(AP, AB)

    # 返回距离
    distance = np.linalg.norm(cross_product) / np.linalg.norm(AB)

    return distance


def point_to_vector_distance(P, v):
    # 计算叉积 P x v
    cross_product = np.cross(P, v)

    # 计算叉积的模长
    cross_norm = np.linalg.norm(cross_product)

    # 计算向量 v 的模长
    v_norm = np.linalg.norm(v)

    # 计算点到向量的最短距离
    distance = cross_norm / v_norm
    return distance


class DRLEnvironment(object):
    action_space = (4,)
    max_axis_velocity = 2.0

    def __init__(
            self,
            drone_name="drone_1",
            viz_image_cv2=True,
            observation_type="images"
    ):
        self.start_position = None
        self.first_time_training = True
        self.last_position = None
        self.start_time = None
        self.gate_pass_time = None
        self.gate_facing = []
        self.gate_poses_ground_truth = []
        self.drone_name = drone_name
        self.viz_image_cv2 = viz_image_cv2

        self.last_angle = 0

        self.airsim_client = airsim.MultirotorClient()
        self.airsim_client_images = airsim.MultirotorClient()
        self.airsim_client_odom = airsim.MultirotorClient()

        self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS = (
            10  # see https://github.com/microsoft/AirSim-Drone-Racing-Lab/issues/38
        )

        self.observation_type = observation_type

        # reward
        self.max_distance = 100
        self.previous_distance = 2
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False
        self.last_action = np.array([0, 0, 0])

        self.last_sp = 0
        self.last_d = 0

        self.reward_behind_zero_count = 0

        if self.observation_type == "lidar" and self.viz_image_cv2:
            self.vis = o3d.visualization.Visualizer()

        self.is_log_monitor_thread_active = False

    def get_orientation(self, vehicle_name="drone_1"):
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        q = state.kinematics_estimated.orientation
        return np.array([q.x_val, q.y_val, q.z_val, q.w_val])

    def getPitchRollYaw(self, vehicle_name="drone_1"):
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        q = state.kinematics_estimated.orientation
        p, r, y = airsim.to_eularian_angles(q)
        # Get quaternion to calculate rotation angle in Z axis (yaw)
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                           1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        return p, r, y, angle

    def calculatePitchRollYaw(self, q):
        p, r, y = airsim.to_eularian_angles(q)
        # Get quaternion to calculate rotation angle in Z axis (yaw)
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                           1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        return p, r, y, angle

    # 计算从当前位置到目标位置的方向。
    # 计算无人机当前位置和目标位置之间的方向差，并考虑当前航向的影响。通过计算角度差，可以帮助无人机判断需要调整的方向角度，从而朝向目标飞行。
    def goal_direction(self, pos, target_pos):
        pitch, roll, yaw, angle = self.getPitchRollYaw()
        yaw = math.degrees(yaw)
        pos_angle = math.atan2(target_pos[1] - pos[1], target_pos[0] - pos[0])
        pos_angle = math.degrees(pos_angle) % 360
        track = math.radians(pos_angle - yaw)
        # return ((math.degrees(track) - 180) % 360) - 180
        return np.array([track])

    def get_linear_acceleration(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        temp_vel[0] = state.kinematics_estimated.linear_acceleration.x_val
        temp_vel[1] = state.kinematics_estimated.linear_acceleration.y_val
        temp_vel[2] = state.kinematics_estimated.linear_acceleration.z_val
        return temp_vel

    def get_velocity(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        temp_vel[0] = state.kinematics_estimated.linear_velocity.x_val
        temp_vel[1] = state.kinematics_estimated.linear_velocity.y_val
        temp_vel[2] = state.kinematics_estimated.linear_velocity.z_val
        return temp_vel

    def get_angle_velocity(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        temp_vel[0] = state.kinematics_estimated.angular_velocity.x_val
        temp_vel[1] = state.kinematics_estimated.angular_velocity.y_val
        temp_vel[2] = state.kinematics_estimated.angular_velocity.z_val
        return temp_vel

    def get_speed(self, vehicle_name="drone_1"):
        temp_val = self.get_velocity(vehicle_name=vehicle_name)
        v = np.linalg.norm(temp_val)
        return v

    def get_w_speed(self, vehicle_name="drone_1"):
        temp_val = self.get_angle_velocity(vehicle_name=vehicle_name)
        v = np.linalg.norm(temp_val)
        return v

    def get_angular_velocity_velocity(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        temp_vel[0] = state.kinematics_estimated.angular_velocity.x_val
        temp_vel[1] = state.kinematics_estimated.angular_velocity.y_val
        temp_vel[2] = state.kinematics_estimated.angular_velocity.z_val
        return temp_vel

    def get_position(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        temp_vel[0] = state.kinematics_estimated.position.x_val
        temp_vel[1] = state.kinematics_estimated.position.y_val
        temp_vel[2] = state.kinematics_estimated.position.z_val
        return temp_vel

    def get_true_position(self, vehicle_name="drone_1"):
        temp_vel = np.array([0.0, 0.0, 0.0])
        state = self.airsim_client.simGetGroundTruthKinematics(vehicle_name=vehicle_name)
        temp_vel[0] = state.position.x_val
        temp_vel[1] = state.position.y_val
        temp_vel[2] = state.position.z_val
        return temp_vel

    def get_observation_space(self):
        if self.observation_type == "images":
            return (3, 240, 320)
        elif self.observation_type == "god":
            return (16, 1)
        else:
            return (3000, 3)

    def start_log_monitor_callback_thread(self):
        if not self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = True

            self.log_monitor_callback_thread = threading.Thread(
                target=self.repeat_log_monitor_callback
            )

            self.log_monitor_callback_thread.start()

    def stop_log_monitor_callback_thread(self):
        if self.is_log_monitor_thread_active:
            self.is_log_monitor_thread_active = False

    def open_log_file(self):
        path = r"D:\GraduationDesign\AirSim-Drone-Racing-Lab-windowsue4\ADRL\ADRL\Saved\Logs\RaceLogs"
        # path = "/home/JorgeGonzalez/ADRL/ADRL/ADRL/Saved/Logs/RaceLogs"
        files = os.listdir(path)
        list_of_files = [os.path.join(path, basename) for basename in files if basename.endswith(".log")]
        latest_file = max(list_of_files, key=os.path.getctime)
        return open(latest_file, "r+")

    def follow_log_file(self, filename):
        filename.seek(0, 2)
        while self.is_log_monitor_thread_active:
            line = filename.readline()
            if not line:
                time.sleep(0.25)
                continue
            yield line

    def check_colission(self, line):
        tokens = line.split()
        # print(line)
        if tokens[0] == self.drone_name and tokens[3] == "penalty":
            if int(tokens[4]) > 0:
                self.has_collission = True

        if tokens[0] == self.drone_name and tokens[3] == "finished":
            self.has_finished = True

    def repeat_log_monitor_callback(self):
        f = self.open_log_file()
        for line in self.follow_log_file(f):
            self.check_colission(line)

    def calculate_safety_reward(self, drone_pos, gate_pos, gate_size=None, d_max=5):
        if gate_size is None:
            gate_size = np.array([3, 2])
        from scipy.spatial.transform import Rotation as R
        x_drone, y_drone, z_drone = drone_pos[0], drone_pos[1], drone_pos[2]
        # 门位置
        x_gate, y_gate, z_gate = gate_pos[0], gate_pos[1], gate_pos[2]
        gate_facing = self.gate_facing[self.next_gate]
        gate_facing = np.array([gate_facing.x_val, gate_facing.y_val, gate_facing.z_val])
        gate_facing_vector = gate_facing + gate_pos
        d = point_to_vector_distance(drone_pos, gate_facing_vector)
        r_s = d
        return r_s

    def calculate_reward(self):
        # 获取无人机的状态
        drone_state = self.airsim_client_odom.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        linear_velocity = drone_state.kinematics_estimated.linear_velocity

        # print("position,",self.get_position())

        reward = 0
        distance = 0
        done = False

        wv = self.get_w_speed()
        # print("angular speed",wv)
        reward_wv = -(wv ** 2) * 0.1
        reward += reward_wv

        speed = self.get_speed()
        reward_v = speed * 0.1
        reward += reward_v
        if speed < 0.8:
            reward -= 0.5
        # 获取当前通过的门
        lastGatePassed = self.airsim_client_odom.simGetLastGatePassed(self.drone_name)  # 0 1 2 3……

        if lastGatePassed > 100:
            lastGatePassed = -1
        gate = self.gate_poses_ground_truth[self.next_gate]
        drone_position = np.array([position.x_val, position.y_val, position.z_val])
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
        distance = position.distance_to(gate.position)
        # 判断是否通过了预定门
        if lastGatePassed == self.next_gate:
            reward += 15 - 10 * distance
            self.next_gate += 1
            self.last_sp = 0
            self.last_d = 0
            self.previous_distance = self.max_distance
            gate = self.gate_poses_ground_truth[self.next_gate]
            gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
            distance = position.distance_to(gate.position)
        else:
            reward += -1
            if lastGatePassed > self.next_gate:
                done = True
            # 计算门距离和方向
            if lastGatePassed < len(self.gate_poses_ground_truth) - 1:
                if self.next_gate == 0:
                    last_gate_position = self.start_position
                else:
                    last_gate = self.gate_poses_ground_truth[self.next_gate - 1]
                    last_gate_position = np.array(
                        [last_gate.position.x_val, last_gate.position.y_val, last_gate.position.z_val])
                # 计算投影长度
                vector_AB = drone_position - last_gate_position
                vector_BC = gate_position - last_gate_position
                # 计算投影长度
                sp = np.dot(vector_AB, vector_BC) / (np.linalg.norm(vector_BC) + 0.001)
                # 点到直线
                d = point_to_line_distance(drone_position, last_gate_position, gate_position)
                reward_d = -(d - self.last_d) - 0.1 - d * 0.1
                reward += reward_d
                distance = position.distance_to(gate.position)
                if self.previous_distance > distance:
                    reward += 0.5
                    if d <= 0.5 and abs(self.last_d - d) < 0.3:
                        reward += 1
                else:
                    reward += -1.5
                self.last_d = d
                # print("self.last_sp:",self.last_sp,end=" || ")

                reward_p = 0
                if sp > np.linalg.norm(vector_BC) + 0.001:
                    reward_p = -(sp - self.last_sp - 0.001)
                else:
                    reward_p = sp - self.last_sp
                reward += reward_p
                self.last_sp = sp
                # print("speed reward:",reward,end=" || ")
                # print("reward_d:",reward_d,end=" || ")
                # print("reward_p:",reward_p,end=" || ")
                # print("reward:", reward)
                # reward_safe = self.calculate_safety_reward(drone_position,gate_position)
                # print("reward_safe:",reward_safe,end=" || ")

                # 如果距离超过最大值，奖励为0且回合结束
                # print(distance)
                # print(drone_position[2])
                self.last_z_distance = abs(self.last_position.z_val - gate_position[2])
                z_distance = abs(drone_position[2] - gate_position[2])
                if z_distance < 0.5:
                    reward_zdistance = 1
                else:
                    reward_zdistance = (self.last_z_distance - z_distance - 0.1) * 3
                    # print('reward_zdistance',reward_zdistance)
                reward += reward_zdistance

                if drone_position[2] > 3.2:
                    reward -= 1

                if distance > 3:
                    direction_vector = gate_position - drone_position
                    direction_vector /= np.linalg.norm(direction_vector) + 0.0001
                    v = self.get_velocity()
                    v /= np.linalg.norm(v) + 0.0001
                    direction_vector_magnitude = np.linalg.norm(direction_vector)
                    v_magnitude = np.linalg.norm(v)
                    similar = (np.dot(direction_vector, v) / (v_magnitude * direction_vector_magnitude + 1e-4))
                    reward += similar ** 2

                if distance > self.max_distance:
                    reward = -2  # 可以根据需求调整
                    done = True

                self.previous_distance = distance

                if reward < 0.25:
                    reward -= 0.3
        # 查是否取消资格、发生碰撞或完成赛道
        isDisqualified = self.airsim_client_odom.simIsRacerDisqualified(self.drone_name)
        if isDisqualified:
            reward = 0  # 可以根据需求调整
            done = True

        if self.has_collission:
            reward += -2  # 可以根据需求调整
            done = True

        if self.has_finished:
            reward = 0
            done = True

        position_move = position.distance_to(self.last_position)
        self.last_position = position

        if abs(position_move) < 0.1 and self.next_gate > 1:
            print("move too slow", position_move)
            reward = -5
            done = True
        # 时间超过5分钟，回合结束
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 100:
            reward += 0  # 可以根据需求调整
            done = True
        if reward < 0:
            self.reward_behind_zero_count += 1
        else:
            self.reward_behind_zero_count -= 0.5
        if self.reward_behind_zero_count >= 15:
            reward += -5
            done = True

        return (reward, done)

    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):

        # self.start_image_callback_thread()

        self.airsim_client.simStartRace(tier)

        self.gate_pass_time = time.time()
        self.start_time = time.time()
        self.start_log_monitor_callback_thread()

        self.initialize_drone()
        self.takeoff()

        time.sleep(1)

        self.last_position = self.airsim_client.getMultirotorState(
            vehicle_name=self.drone_name).kinematics_estimated.position
        self.start_position = self.get_position()

        if self.first_time_training:
            self.first_time_training = False
            self.get_ground_truth_gate_poses()
            self.get_ground_truth_gate_facing()
        drone_state = self.airsim_client_odom.getMultirotorState()
        # Get quaternion to calculate rotation angle in Z axis (yaw)
        q = drone_state.kinematics_estimated.orientation
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val),
                           1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        self.last_angle = angle
        self.last_sp = 0
        self.last_d = 0
        self.reward_behind_zero_count = 0
        self.last_action = np.array([0, 0, 0])
        return self.get_observation()

    def initialize_drone(self):
        self.airsim_client.enableApiControl(vehicle_name=self.drone_name)
        self.airsim_client.arm(vehicle_name=self.drone_name)

        # set default values for trajectory tracker gains
        traj_tracker_gains = airsim.TrajectoryTrackerGains(
            kp_cross_track=5.0,
            kd_cross_track=0.0,
            kp_vel_cross_track=3.0,
            kd_vel_cross_track=0.0,
            kp_along_track=0.4,
            kd_along_track=0.0,
            kp_vel_along_track=0.04,
            kd_vel_along_track=0.0,
            kp_z_track=2.0,
            kd_z_track=0.0,
            kp_vel_z=0.4,
            kd_vel_z=0.0,
            kp_yaw=3.0,
            kd_yaw=0.1,
        )

        self.airsim_client.setTrajectoryTrackerGains(
            traj_tracker_gains, vehicle_name=self.drone_name
        )
        time.sleep(0.2)

    def takeoff(self, takeoff_height=1.0):
        start_position = self.airsim_client.simGetVehiclePose(
            vehicle_name=self.drone_name
        ).position
        takeoff_waypoint = airsim.Vector3r(
            start_position.x_val,
            start_position.y_val,
            start_position.z_val - takeoff_height,
        )

        self.airsim_client.moveOnSplineAsync(
            [takeoff_waypoint],
            vel_max=15.0,
            acc_max=5.0,
            add_position_constraint=True,
            add_velocity_constraint=False,
            add_acceleration_constraint=False,
            vehicle_name=self.drone_name,
        ).join()

    def get_ground_truth_gate_poses(self):
        gate_names_sorted_bad = sorted(self.airsim_client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is of the form `GateN_GARBAGE`. for example:
        # ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # we sort them by their ibdex of occurence along the race track(N), and ignore the unreal garbage number after the underscore(GARBAGE)
        gate_indices_bad = [
            int(gate_name.split("_")[0][4:]) for gate_name in gate_names_sorted_bad
        ]
        gate_indices_correct = sorted(
            range(len(gate_indices_bad)), key=lambda k: gate_indices_bad[k]
        )
        gate_names_sorted = [
            gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct
        ]
        self.gate_poses_ground_truth = []
        for gate_name in gate_names_sorted:
            curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            counter = 0
            while (
                    math.isnan(curr_pose.position.x_val)
                    or math.isnan(curr_pose.position.y_val)
                    or math.isnan(curr_pose.position.z_val)
            ) and (counter < self.MAX_NUMBER_OF_GETOBJECTPOSE_TRIALS):
                # print(f"DEBUG: {gate_name} position is nan, retrying...")
                counter += 1
                curr_pose = self.airsim_client.simGetObjectPose(gate_name)
            assert not math.isnan(
                curr_pose.position.x_val
            ), f"ERROR: {gate_name} curr_pose.position.x_val is still {curr_pose.position.x_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.y_val
            ), f"ERROR: {gate_name} curr_pose.position.y_val is still {curr_pose.position.y_val} after {counter} trials"
            assert not math.isnan(
                curr_pose.position.z_val
            ), f"ERROR: {gate_name} curr_pose.position.z_val is still {curr_pose.position.z_val} after {counter} trials"
            self.gate_poses_ground_truth.append(curr_pose)
            # print(gate_name, curr_pose.position)

    # 这是一个实用函数，用于根据四元数计算门的朝向向量，这个向量可以用来给 moveOnSplineVelConstraints() 设置速度约束
    # "scale" 参数控制向量的缩放比例，从而影响约束的速度
    def get_gate_facing_vector_from_quaternion(self, airsim_quat, scale=1.0):
        import numpy as np
        q = np.array(
            [
                airsim_quat.w_val,  # 四元数的 w 分量
                airsim_quat.x_val,  # 四元数的 x 分量
                airsim_quat.y_val,  # 四元数的 y 分量
                airsim_quat.z_val,  # 四元数的 z 分量
            ],
            dtype=np.float64,
        )
        n = np.dot(q, q)  # 计算四元数的模的平方
        if n < np.finfo(float).eps:  # 如果模非常小（接近零），返回默认的向量
            return airsim.Vector3r(0.0, 1.0, 0.0)

        # 归一化四元数，确保它是单位四元数
        q *= np.sqrt(2.0 / n)
        q = np.outer(q, q)  # 计算外积，得到一个 4x4 的矩阵

        # 使用四元数外积生成旋转矩阵
        rotation_matrix = np.array(
            [
                [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]],
            ]
        )

        # 旋转矩阵的第二列是旋转后表示物体朝向的向量（即门的朝向）
        gate_facing_vector = rotation_matrix[:, 1]

        # 返回缩放后的朝向向量
        return airsim.Vector3r(
            scale * gate_facing_vector[0],
            scale * gate_facing_vector[1],
            scale * gate_facing_vector[2],
        )

    def init_race_environment(self):
        self.airsim_client.confirmConnection()
        self.airsim_client_images.confirmConnection()
        self.airsim_client_odom.confirmConnection()
        self.load_level(level_name="Soccer_Field_Medium")

        if self.observation_type == "lidar" and self.viz_image_cv2:
            self.vis.create_window(width=640, height=480)

    def get_observation(self):
        if self.observation_type == "images":
            return self.get_camera_image()
        elif self.observation_type == "god":
            return self.get_god_state()
        else:
            return self.get_lidar_points()

    def get_camera_image(self):
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)

        # if self.viz_image_cv2:
        # cv2.imshow("Drone FPV", img_rgb)
        # cv2.waitKey(1)
        # img_rgb = np.moveaxis(img_rgb, [2], [0])
        # print(img_rgb.shape)#240 320 3

        return img_rgb

    def get_deep_image(self):
        responses = self.airsim_client.simGetImages([
            airsim.ImageRequest('front_center', airsim.ImageType.DepthVis, True, False)])
        img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
        # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
        img_depth_vis = img_depth_planar / 100
        img_depth_vis[img_depth_vis > 1] = 1.
        # 3. 转换为整形
        img_depth_vis = (img_depth_vis * 255).astype(np.uint8)

        return img_depth_vis

    def get_lidar_points(self):
        lidar_data = self.airsim_client_odom.getLidarData(lidar_name="LidarSensor1", vehicle_name=self.drone_name)

        if self.viz_image_cv2:
            request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
            response = self.airsim_client_images.simGetImages(request)
            img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)

        complete_points = np.zeros(self.get_observation_space())
        points = np.array(lidar_data.point_cloud, dtype=np.dtype('f4'))

        # if self.get_observation_space()[0] < len(points):
        #      print(len(points))

        if (len(points)) < 3:
            return complete_points

        points = np.reshape(points, (int(points.shape[0] / 3), 3))
        points_shape = np.shape(points);

        complete_points[:points_shape[0], :points_shape[1]] = points

        if self.viz_image_cv2:
            self.vis.clear_geometries()
            template_ = o3d.geometry.PointCloud()
            template_.points = o3d.utility.Vector3dVector(complete_points)
            self.vis.add_geometry(template_)
            ctr = self.vis.get_view_control()
            ctr.rotate(-200, 0)

            self.vis.poll_events()
            self.vis.update_renderer()

            cv2.imshow("Drone FPV", img_rgb)
            cv2.waitKey(1)
        # print(complete_points.shape)
        # print("386 complete point",complete_points.shape) (3000, 3)
        return complete_points

    def get_god_state(self):
        # 获取无人机的状态
        self.next_gate = min(self.next_gate, len(self.gate_poses_ground_truth) - 1)

        linear_velocity = self.get_velocity()
        angular_velocity = self.get_angular_velocity_velocity()
        linear_acceleration = self.get_linear_acceleration()
        orientation = self.get_orientation()
        drone_pitch, drone_roll, drone_yaw, drone_angle = self.getPitchRollYaw()

        gate = self.gate_poses_ground_truth[self.next_gate]
        gate_pitch, gate_roll, gate_yaw, gate_angle = self.calculatePitchRollYaw(gate.orientation)

        drone_position = self.get_position()
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])

        # goal_direction = self.goal_direction(position, gate_position)
        relative_pos = gate_position - drone_position
        relative_yaw = drone_yaw - gate_yaw
        # print(relative_yaw)

        # relative_pos = relative_pos / np.linalg.norm(relative_pos)
        with open("state.txt", "a") as f:
            f.write(f"{self.next_gate},{relative_pos}\n")

        # 目标方向:2、相对距离:3、俯仰姿态:3、目标方向：1、相对于世界坐标的偏转角度:1
        # state = np.concatenate((drone_position,orientation,gate_position))

        state = np.concatenate((linear_velocity, linear_acceleration, angular_velocity, orientation, relative_pos))
        # state = np.concatenate((drone_position,gate_position))
        # state = relative_pos
        # state = np.concatenate((relative_pos,np.array([relative_yaw])))
        # img = self.get_camera_image()
        # print(relative_yaw)
        # if 1< relative_yaw <3:
        #     cv2.imwrite(f'vae_train_imgs/{relative_pos[0]}-{relative_pos[1]}-{relative_pos[2]}-{relative_yaw}.jpg',img)

        return state

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset(self):
        self.airsim_client.simResetRace()
        self.stop_log_monitor_callback_thread()
        # self.stop_image_callback_thread()

        # if self.observation_type == "lidar":
        #     self.vis.destroy_window()

        self.previous_distance = 2
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False

    def step(self, action):

        # print(action)
        # x = np.clip(action[0], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float)
        # y = np.clip(action[1], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float)
        a = action[0].astype(np.float64)
        b = action[1].astype(np.float64)
        c = action[2].astype(np.float64)
        d = action[3].astype(np.float64)
        # print(direction_vector)
        self.airsim_client.moveByVelocityZAsync(x
                                                , y
                                                , z=gate.position.z_val - 0.5
                                                , duration=0.2
                                                , drivetrain=airsim.DrivetrainType.ForwardOnly
                                                , yaw_mode=airsim.YawMode(is_rate=False)
                                                , vehicle_name=self.drone_name)
        # self.airsim_client.moveByMotorPWMsAsync(a,b,c,d,duration=0.1,vehicle_name=self.drone_name)

        time.sleep(0.05)
        done = False
        (reward, done) = self.calculate_reward()

        # action_reward = np.linalg.norm(action-self.last_action)
        # self.last_action = action

        # print(z,reward)
        return self.get_observation(), reward, done, self.next_gate

    def get_ground_truth_gate_facing(self):
        self.gate_facing = [self.get_gate_facing_vector_from_quaternion(gatepose.orientation)
                            for gatepose in self.gate_poses_ground_truth]


