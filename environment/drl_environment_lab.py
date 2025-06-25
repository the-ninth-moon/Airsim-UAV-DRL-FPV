import PIL
import airsimdroneracinglab as airsim
import matplotlib
import numpy as np
import cv2
import threading
import time
import math
import os

import torch
from gymnasium import spaces
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
from cvae.cvae import CVAE, Config  # 假设模型定义在 model.py 文件中
import gymnasium as gym
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from depth_anything_v2.dpt import DepthAnythingV2

def point_to_line_distance(P, A, B):
    # 向量 AB 和 AP
    AB = B - A
    AP = P - A

    # 计算叉积
    cross_product = np.cross(AP, AB)

    # 返回距离
    distance = np.linalg.norm(cross_product) / np.linalg.norm(AB)

    return distance
#计算相对姿态
def quaternion_multiply(q1, q2):
    x1, y1, z1,w1 = q1
    x2, y2, z2,w2 = q2
    return [
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 + y1*w2 + z1*x2 - x1*z2,
        w1*z2 + z1*w2 + x1*y2 - y1*x2
    ]

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

def body_to_global_velocity(vx_body, vy_body, vz_body, orientation):
    pitch, roll, yaw = airsim.to_eularian_angles(orientation)

    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    R = [
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r]
    ]

    # 将机体坐标系下的速度转换为全局坐标系下的速度
    vx_global = R[0][0] * vx_body + R[0][1] * vy_body + R[0][2] * vz_body
    vy_global = R[1][0] * vx_body + R[1][1] * vy_body + R[1][2] * vz_body
    vz_global = R[2][0] * vx_body + R[2][1] * vy_body + R[2][2] * vz_body

    return vx_global, vy_global, vz_global

def global_to_body_point(relative_position_global, orientation):
    """
    将世界坐标系下的点转换到无人机机体坐标系下。

    Args:
        target_pos_global: 目标点在世界坐标系中的坐标 (airsim.Vector3r).
        drone_pos_global: 无人机在世界坐标系中的坐标 (airsim.Vector3r).
        orientation: 无人机的姿态 (airsim.Quaternionr).

    Returns:
        target_pos_body: 目标点在无人机机体坐标系中的坐标 (airsim.Vector3r).
    """

    # 1. 计算世界坐标系下，目标点相对于无人机的向量
    vector_global_list = relative_position_global
    pitch, roll, yaw = airsim.to_eularian_angles(orientation)  # 正确的顺序
    cos_r = math.cos(roll)
    sin_r = math.sin(roll)
    cos_p = math.cos(pitch)
    sin_p = math.sin(pitch)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)

    R = [
        [cos_y * cos_p, cos_y * sin_p * sin_r - sin_y * cos_r, cos_y * sin_p * cos_r + sin_y * sin_r],
        [sin_y * cos_p, sin_y * sin_p * sin_r + cos_y * cos_r, sin_y * sin_p * cos_r - cos_y * sin_r],
        [-sin_p, cos_p * sin_r, cos_p * cos_r]
    ]

    # 4. 计算旋转矩阵的转置矩阵 R_transpose (用于 global to body)
    R_transpose = [
        [R[0][0], R[1][0], R[2][0]],
        [R[0][1], R[1][1], R[2][1]],
        [R[0][2], R[1][2], R[2][2]]
    ]

    # 5. 将世界坐标系下的相对向量转换到机体坐标系
    vector_body_list = [0, 0, 0]
    for i in range(3):
        for j in range(3):
            vector_body_list[i] += R_transpose[i][j] * vector_global_list[j]


    return np.array(vector_body_list)

class DRLEnvironment(gym.Env):

    action_space = (4,)
    max_axis_velocity = 2.0
    
    def __init__(
        self,
        drone_name="drone_1",
        viz_image_cv2=True,
        observation_type="images",
        show_img = False
    ):

        self.img_state = None
        self.last_center_difference = 0
        self.last_image_area = 0
        self.img_reward = None
        self.final_img = None
        self.depth_img = None
        self.camer_Image = None
        self.is_image_thread_active = False
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
        self.show_img = show_img
        
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
        self.gate_detect_model = YOLO(f'D:/GraduationDesign/gate-detect/runs/detect/train/weights/best.pt')  # 加载你的YOLOv8n模型

        self.vae_model=CVAE().to('cuda')
        self.vae_model.load_state_dict(torch.load(f'D:/GraduationDesign/airsim-drl-drone-racing-lab - IMAGE/cvae/cvae_output_onlyimginfo/best_cvae.pth'))
        self.vae_model.eval()  # 设置为评估模式
        self.cvae_transform = transforms.Compose([
            transforms.Resize(Config.img_size),
            transforms.ToTensor(),  # 不需要 RandomHorizontalFlip() 等数据增强
        ])

        # 加载深度预测模型
        deep_model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        }
        self.depth_model = DepthAnythingV2(**deep_model_configs['vits'])
        self.depth_model.load_state_dict(torch.load(f'depth_model_checkpoints/depth_anything_v2_vits.pth', map_location='cpu'))
        self.depth_model = self.depth_model.to('cuda').eval()

        self.last_sp = 0
        self.last_d = 0

        self.reward_behind_zero_count = 0
        
        # if self.observation_type == "lidar" and self.viz_image_cv2:
        #     self.vis = o3d.visualization.Visualizer()
            
        self.is_log_monitor_thread_active = False
        self.action_dim = 4
        self.action_space = spaces.Box(
            low=np.array([-5,-3,-5]),
            high=np.array([5,3,5]),
            dtype=np.float64
        ) #连续
        # self.action_space = spaces.Box(
        #     low=np.array([0,-5,-8]),
        #     high=np.array([15,5,8]),
        #     dtype=np.float64
        # ) #连续
        if self.action_dim==4:
            self.action_space = spaces.Box(
                low=np.array([0,-3,-8,-3]),
                high=np.array([15,3,8,3]),
                dtype=np.float64
            ) #连续

        # self.action_space = spaces.MultiDiscrete([5,5,5])

        # self.action_space = spaces.Box(
        #     low=np.array([-1,-1,-1.5,0.1]),
        #     high=np.array([1,1,1.5,0.8]),
        #     dtype=np.float64
        # ) #连续
        # self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7+64,), dtype=np.float32)
        if self.observation_type == "raw_image":
            self.vector_observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
            #
            # 图像空间的定义 (240, 320, 1)  假设像素值范围是 0-255，类型为 uint8
            self.image_observation_space = spaces.Box(low=0, high=255, shape=(240, 320, 3), dtype=np.uint8)
            #
            # # 复合观察空间：使用 Dict 组合向量和图像空间
            self.observation_space = spaces.Dict({
                'vector': self.vector_observation_space,
                'image': self.image_observation_space
            })


        # 启动两个线程，用于定时调用图像和姿态更新回调函数
        self.image_callback_thread = threading.Thread(
            target=self.repeat_timer_image_callback, args=(self.image_callback, 0.02)  # 每30ms调用一次
        )
        self.init_race_environment()  # 初始化环境

        self.last_action = np.array([0] * self.action_dim)
    def start_image_callback_thread(self):
        # 启动图像回调的后台线程
        if not self.is_image_thread_active:
            self.is_image_thread_active = True
            self.image_callback_thread.start()  # 启动图像回调线程
            print("Started image callback thread")  # 打印信息表示线程已启动

    def stop_image_callback_thread(self):
        # 停止图像回调线程
        if self.is_image_thread_active:
            # 设置标志位为False，表示线程应停止
            self.is_image_thread_active = False
            # 等待图像回调线程结束
            self.image_callback_thread.join()
            # 打印信息表示线程已停止
            print("Stopped image callback thread.")
    def repeat_timer_image_callback(self, task, period):
        # 定期每隔 "period" 秒执行一次图像回调
        while self.is_image_thread_active:
            task()  # 执行图像回调任务
            time.sleep(period)  # 等待指定时间间隔

    def get_observation_space(self):
        if self.observation_type == "images":
            return  (3, 240, 320)
        elif self.observation_type == "god":
            return (19,1)
        else:
            return (3000, 3)
    def get_orientation(self, vehicle_name="drone_1"):
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        q = state.kinematics_estimated.orientation
        return np.array([q.x_val, q.y_val, q.z_val, q.w_val])
    def getPitchRollYaw(self, vehicle_name="drone_1"):
        state = self.airsim_client.getMultirotorState(vehicle_name=vehicle_name)
        q = state.kinematics_estimated.orientation
        p,r,y = airsim.to_eularian_angles(q)
        #Get quaternion to calculate rotation angle in Z axis (yaw)
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val) , 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        return p,r,y,angle
    def calculatePitchRollYaw(self, q):
        p,r,y = airsim.to_eularian_angles(q)
        #Get quaternion to calculate rotation angle in Z axis (yaw)
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val) , 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        return p,r,y,angle

    # 计算从当前位置到目标位置的方向。
    # 计算无人机当前位置和目标位置之间的方向差，并考虑当前航向的影响。通过计算角度差，可以帮助无人机判断需要调整的方向角度，从而朝向目标飞行。
    def goal_direction(self, pos,target_pos):
        pitch, roll, yaw,angle = self.getPitchRollYaw()
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
        #path = "/home/JorgeGonzalez/ADRL/ADRL/ADRL/Saved/Logs/RaceLogs"
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
        #print(line)
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
            gate_size = np.array([3,2])
        from scipy.spatial.transform import Rotation as R
        x_drone, y_drone, z_drone = drone_pos[0],drone_pos[1],drone_pos[2]
        # 门位置
        x_gate, y_gate, z_gate = gate_pos[0],gate_pos[1],gate_pos[2]
        gate_facing = self.gate_facing[self.next_gate]
        gate_facing = np.array([gate_facing.x_val,gate_facing.y_val,gate_facing.z_val])
        gate_facing_vector = gate_facing+gate_pos
        d = point_to_vector_distance(drone_pos, gate_facing_vector)
        r_s = d
        return r_s
    def calculate_safety_reward(self, drone_pos, gate_pos, gate_size=None, d_max=5):
        if gate_size is None:
            gate_size = np.array([3,2])
        from scipy.spatial.transform import Rotation as R
        x_drone, y_drone, z_drone = drone_pos[0],drone_pos[1],drone_pos[2]
        # 门位置
        x_gate, y_gate, z_gate = gate_pos[0],gate_pos[1],gate_pos[2]
        gate_facing = self.gate_facing[self.next_gate]
        gate_facing = np.array([gate_facing.x_val,gate_facing.y_val,gate_facing.z_val])
        gate_facing_vector = gate_facing+gate_pos
        d = point_to_vector_distance(drone_pos, gate_facing_vector)
        r_s = d
        return r_s
    def calculate_reward_old(self,verbose):
        # 获取无人机的状态
        drone_state = self.airsim_client_odom.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        linear_velocity = drone_state.kinematics_estimated.linear_velocity

        # print("position,",self.get_position())

        reward = 0
        distance = 0
        done = False
        truncated = False

        wv = self.get_w_speed()
        # print("angular speed",wv)
        reward_wv = -(wv**2) * 0.3
        reward += reward_wv

        speed = self.get_speed()
        reward_v = -(speed-1)*(speed-5)*0.2
        reward += reward_v
        if speed<0.8 or speed>5:
            reward -= 1
            if speed<0.5:
                reward -= 3
        # 获取当前通过的门
        lastGatePassed = self.airsim_client_odom.simGetLastGatePassed(self.drone_name) #0 1 2 3……

        if lastGatePassed > 100:
            lastGatePassed = -1
        gate = self.gate_poses_ground_truth[self.next_gate]
        drone_position = np.array([position.x_val, position.y_val, position.z_val])
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val-0.8])
        distance = position.distance_to(gate.position)
        # 判断是否通过了预定门
        if lastGatePassed == self.next_gate:
            if self.next_gate==len(self.gate_poses_ground_truth)-1:
                print("Finished!!!!")
                return (300,True,False)
            reward += (15-6*distance)*4
            self.next_gate += 1
            self.last_sp = 0
            self.last_d = 0
            self.previous_distance = self.max_distance
            gate = self.gate_poses_ground_truth[self.next_gate]
            gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val-0.8])
            distance = position.distance_to(gate.position)
            self.reward_behind_zero_count = 0
            if verbose:
                print(f"Gate: {self.next_gate}| Speed: {speed:.2f} | Reward: {reward:.2f}")
            return reward,False,False
        else:
            reward += 0
            if lastGatePassed > self.next_gate:
                print("done for strange reason")
                done = True
            # 计算门距离和方向
            if lastGatePassed < len(self.gate_poses_ground_truth) - 1:
                if self.next_gate == 0:
                    last_gate_position = self.start_position
                else:
                    last_gate = self.gate_poses_ground_truth[self.next_gate-1]
                    last_gate_position = np.array([last_gate.position.x_val, last_gate.position.y_val, last_gate.position.z_val-0.8])
                # 计算投影长度
                vector_AB = drone_position - last_gate_position
                vector_BC = gate_position - last_gate_position
                # 计算投影长度
                sp = np.dot(vector_AB, vector_BC) / (np.linalg.norm(vector_BC)+0.001)

                #点到直线
                d = point_to_line_distance(drone_position, last_gate_position,gate_position)
                distance = position.distance_to(gate.position)
                reward_d = 0
                if self.previous_distance>distance:
                    if not self.previous_distance==self.max_distance:
                        reward_d += (self.previous_distance-distance)*2+1
                        # print(f"distance_change:{self.previous_distance-distance:.2f}")
                        if d<=0.5 and abs(self.last_d-d)<0.3:
                            reward_d += 1
                else:
                    reward_d += (self.previous_distance-distance)-1
                self.last_d = d
                # print("self.last_sp:",self.last_sp,end=" || ")

                reward_p = 0
                if sp > np.linalg.norm(vector_BC) + 0.001:
                    reward_p = -(sp - self.last_sp-0.001)
                else:
                    reward_p = sp - self.last_sp
                reward_p *= 6
                reward += reward_p
                reward += reward_d
                self.last_sp = sp
                # print("speed reward:",reward,end=" || ")
                # print("reward_d:",reward_d,end=" | ")
                # print("reward_p:",reward_p,end=" | ")
                # print("reward:", reward)
                # reward_safe = self.calculate_safety_reward(drone_position,gate_position)
                # print("reward_safe:",reward_safe,end=" || ")

                # 如果距离超过最大值，奖励为0且回合结束
                # print(distance)
                # print(drone_position[2])
                self.last_z_distance = abs(self.last_position.z_val-gate_position[2])
                z_distance = abs(drone_position[2]-gate_position[2])
                if (self.last_z_distance-z_distance)>0:
                    reward_zdistance = 0.5+(self.last_z_distance-z_distance)*2
                elif z_distance<0.3:
                    reward_zdistance = 1.5
                else:
                    reward_zdistance =-0.3+(self.last_z_distance-z_distance)*2
                # print('reward_zdistance',reward_zdistance,end=" | ")
                reward += reward_zdistance


                if distance>1.5:
                    direction_vector = gate_position - drone_position
                    direction_vector /= np.linalg.norm(direction_vector) + 0.0001
                    v = self.get_velocity()
                    v /= np.linalg.norm(v) + 0.0001
                    direction_vector_magnitude = np.linalg.norm(direction_vector)
                    v_magnitude = np.linalg.norm(v)
                    similar = (np.dot(direction_vector, v) / (v_magnitude * direction_vector_magnitude+1e-4))
                    print("reward_similar", (similar) * (distance*2),end=" | ")
                    reward += (similar*1.5-0.6) * (distance*2)

                if distance > self.max_distance or z_distance>8:
                    reward = -10 # 可以根据需求调整
                    print("done for too far")
                    done = True

                self.previous_distance = distance

        # 查是否取消资格、发生碰撞或完成赛道
        isDisqualified = self.airsim_client_odom.simIsRacerDisqualified(self.drone_name)
        if isDisqualified:
            reward = 0  # 可以根据需求调整
            done = True

        if self.has_collission:
            reward = -20  # 可以根据需求调整
            done = True
            self.last_position = position
            print("collision!")
            return (reward, done,truncated)

        if self.has_finished:
            reward = 0
            done = True

        position_move = position.distance_to(self.last_position)
        self.last_position = position
        # if abs(position_move)<0.1 and self.next_gate>1:
        #     print("move too slow", position_move)
        #     reward = -5
        #     done = True
        # 时间超过5分钟，回合结束
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 100:
            reward += 0  # 可以根据需求调整
            done = True
            truncated = True
        if reward<0:
            self.reward_behind_zero_count += 1
        else:
            self.reward_behind_zero_count -= 0.6
            self.reward_behind_zero_count = max(0, self.reward_behind_zero_count)
        if self.reward_behind_zero_count >= 60:
            reward += -20
            print("done for too much bad")
            done = True

        reward *= 0.1

        # print("speed:",speed,"discount:",abs((0.01*(speed)*(speed-10))))
        # print("position:",drone_position)
        if verbose:
            print(f"Gate: {self.next_gate}| Speed: {speed:.2f} | Reward: {reward:.2f}")
        return (reward, done,truncated)
    def calculate_reward(self,action=np.array([0,0,0]),verbose=False):
        # 获取无人机的状态
        drone_state = self.airsim_client_odom.getMultirotorState()
        position = drone_state.kinematics_estimated.position
        linear_velocity = drone_state.kinematics_estimated.linear_velocity

        # print("position,",self.get_position())

        reward = 0
        distance = 0
        done = False
        truncated = False

        wv = self.get_w_speed()
        reward_wv = -(wv**2) * 0.05
        reward += reward_wv

        speed = self.get_speed()
        reward_v = -(speed-3)*(speed-10)*0.02
        # print(f"reward_wv:{reward_wv},speed:{speed},speed reward:{reward_v:.2f}")


        reward += reward_v
        # print(f"reward_wv:{reward_wv},speed:{speed},speed reward:{reward_v:.2f}")

        if speed<1.5 or speed>7:
            reward -= 0.6
            if speed<0.5:
                reward -= 1

        reward += reward_v
        # 获取当前通过的门
        lastGatePassed = self.airsim_client_odom.simGetLastGatePassed(self.drone_name) #0 1 2 3……

        if lastGatePassed > 100:
            lastGatePassed = -1
        gate = self.gate_poses_ground_truth[self.next_gate]
        drone_position = np.array([position.x_val, position.y_val, position.z_val])
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val-0.8])
        distance = position.distance_to(gate.position)
        # 判断是否通过了预定门
        if lastGatePassed == self.next_gate:
            if self.next_gate==len(self.gate_poses_ground_truth)-1:
                print("Finished!!!!")
                return (100,True,False)
            reward += (10-5*distance)*(3-speed)*(speed-12)*0.3 if (3<speed<12) else (10-5*distance)*2
            self.next_gate += 1
            self.last_sp = 0
            self.last_d = 0
            self.previous_distance = self.max_distance
            gate = self.gate_poses_ground_truth[self.next_gate]
            gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val-0.8])
            distance = position.distance_to(gate.position)
            self.reward_behind_zero_count = 0
            return reward,False,False
        else:
            reward += 0
            if lastGatePassed > self.next_gate:
                print("done for strange reason")
                done = True
            # 计算门距离和方向
            if lastGatePassed < len(self.gate_poses_ground_truth) - 1:
                gate = self.gate_poses_ground_truth[self.next_gate]
                drone_position = self.get_position()
                gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
                gate_position += np.random.rand(3)
                direction_vector = gate_position - drone_position
                direction_vector /= np.linalg.norm(direction_vector)
                direction_vector *= 5
                state = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name)
                q = state.kinematics_estimated.orientation
                relative_pos_body = global_to_body_point(direction_vector, q)
                v = self.get_velocity()
                v = global_to_body_point(v, q)
                similar = self._calculate_similarity_reward(v,relative_pos_body)
                scaled = ((distance-0.3) / (30 - 0))*3
                # scaled = math.log(distance+1)
                # print(scaled)
                reward += (similar) * scaled
                if distance > self.max_distance:
                    reward = -20  # 可以根据需求调整
                    done = True
                self.previous_distance = distance

        # 查是否取消资格、发生碰撞或完成赛道
        isDisqualified = self.airsim_client_odom.simIsRacerDisqualified(self.drone_name)
        if isDisqualified:
            reward = 0  # 可以根据需求调整
            done = True

        if self.has_collission:
            reward = -15  # 可以根据需求调整
            done = True
            self.last_position = position
            print("collision!")
            return (reward, done,truncated)

        if self.has_finished:
            reward = 0
            done = True

        position_move = position.distance_to(self.last_position)
        self.last_position = position
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 300:
            reward += 0  # 可以根据需求调整
            done = True
            truncated = True
        if reward<0:
            self.reward_behind_zero_count += 1
        else:
            self.reward_behind_zero_count -= 0.6
            self.reward_behind_zero_count = max(0, self.reward_behind_zero_count)
        if self.reward_behind_zero_count >= 60:
            reward = -10
            print("done for too much bad")
            done = True
        # reward *= 2
        print(f"Gate: {self.next_gate}|  Speed:{speed}  | Reward: {reward:.2f}", end="   |  ")
        return (reward, done,truncated)

    def _calculate_similarity_reward(self, action, realv):
        """核心奖励计算逻辑：正则化相似度"""
        a = np.array(action)
        r = realv

        # 正则化处理
        a_norm = np.linalg.norm(a)
        r_norm = np.linalg.norm(r)

        if a_norm < 1e-6 or r_norm < 1e-6:
            return -0.1  # 避免除零，并略微惩罚静止不动? 也可以返回 0

        a_normalized = a / a_norm
        r_normalized = r / r_norm

        # 余弦相似度奖励（方向一致性）， 范围 [-1, 1]
        cosine_similarity = np.dot(a_normalized, r_normalized)

        if a_normalized[2] * r_normalized[2] > 0:
            cosine_similarity += 0
        else:
            # print("wrong z")
            cosine_similarity -= 0.5

        return cosine_similarity  # 奖励范围调整到 [-1, 1]
    def load_level(self, level_name, sleep_sec=2.0):
        self.level_name = level_name
        self.airsim_client.simLoadLevel(self.level_name)
        self.airsim_client.confirmConnection()  # failsafe
        time.sleep(sleep_sec)  # let the environment load completely

    # Starts an instance of a race in your given level, if valid
    def start_race(self, tier=1):
        # self.start_image_callback_thread()
        self.next_gate = 0


        self.airsim_client.simStartRace(tier)
        self.gate_pass_time = time.time()
        self.start_time = time.time()
        self.start_log_monitor_callback_thread()
        self.initialize_drone()
        self.takeoff()

        self.last_position = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name).kinematics_estimated.position
        self.start_position = self.get_position()
        
        if self.first_time_training:
            self.first_time_training = False
            self.get_ground_truth_gate_poses()
            self.get_ground_truth_gate_facing()
        drone_state = self.airsim_client_odom.getMultirotorState()
        #Get quaternion to calculate rotation angle in Z axis (yaw)
        q = drone_state.kinematics_estimated.orientation
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val) , 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))
        self.last_angle = angle
        self.last_sp = 0
        self.last_d = 0
        self.last_center_difference = 0
        self.reward_behind_zero_count = 0
        self.last_action = np.array([0]*self.action_dim)

        gate = self.gate_poses_ground_truth[self.next_gate]
        drone_position = self.get_position()
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val-0.5])
        direction_vector = gate_position - drone_position
        direction_vector /= np.linalg.norm(direction_vector)
        direction_vector *= 5
        self.airsim_client.moveByVelocityAsync(direction_vector[0]
                                                , direction_vector[1]
                                                , direction_vector[2]
                                                , duration = 0.8
                                                , drivetrain= airsim.DrivetrainType.ForwardOnly
                                                , yaw_mode= airsim.YawMode(is_rate=False)
                                                , vehicle_name=self.drone_name)

        return self.get_observation()[0]

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
                #print(f"DEBUG: {gate_name} position is nan, retrying...")
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
            # print(gate_name, curr_pose)
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
        # self.airsim_client_images.confirmConnection()
        self.airsim_client_odom.confirmConnection()
        # self.load_level(level_name="Qualifier_Tier_2")
        self.load_level(level_name="Soccer_Field_Easy")

        if self.observation_type == "lidar" and self.viz_image_cv2:
            self.vis.create_window(width=640,height=480)
        self.start_image_callback_thread()

    def get_observation(self):
        if self.observation_type == "images":    
            return self.get_god_image()[0]
        elif self.observation_type == "god":
            return self.get_god_state()
        elif self.observation_type == "raw_image":
            return self.get_god_state()
        else:
            return self.get_lidar_points()

    def image_callback(self):
        # 获取无人机的FPV相机图像（未压缩的场景图像）
        request = [airsim.ImageRequest("fpv_cam", airsim.ImageType.Scene, False, False)]
        response = self.airsim_client_images.simGetImages(request)

        # 将图像数据转换为1维数组
        img_rgb_1d = np.frombuffer(response[0].image_data_uint8, dtype=np.uint8)
        try:
            # 将1维数组转换为图像（将其重新排列为原始图像的宽、高和颜色通道）
            img_rgb = img_rgb_1d.reshape(response[0].height, response[0].width, 3)
        except:
            print("error in image_callback")
            return
        self.camer_Image = img_rgb
        # YOLOv8检测部分 #############################################
        # 进行推理（YOLOv8会自动处理图像格式）
        results = self.gate_detect_model.predict(img_rgb, verbose=False)  # 禁用控制台输出
        # 绘制检测结果
        if self.show_img:
            annotated_img = img_rgb.copy()
            for result in results:
                # 遍历每个检测到的对象
                for box in result.boxes:
                    # 获取坐标和置信度
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # 转换为整数坐标
                    conf = round(float(box.conf[0]), 2)  # 保留两位小数
                    cls_id = int(box.cls[0])  # 类别ID
                    # 绘制矩形框
                    color = (0, 255, 0)  # BGR格式颜色（这里用绿色）
                    cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                    cv2.rectangle(annotated_img, ((x2+x1)//2, y1), ((x1+x2)//2, y1+2), (255,0,0), cv2.FILLED) #x1右下的点
                    # 构建标签文本（假设你的类别名称为classes）
                    label = f"{self.gate_detect_model.names[cls_id]} {conf * 100:.1f}%"
                    # 计算文本位置
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    # 绘制文本背景
                    cv2.rectangle(annotated_img,
                                  (x1, y1 - 20),
                                  (x1 + tw, y1),
                                  color, -1)
                    # 绘制文本
                    cv2.putText(annotated_img,
                                label,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 255, 255),  # 白色文字
                                2)
        # 深度预测
        with torch.no_grad():
            depth = self.depth_model.infer_image(img_rgb, 518)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_img = depth.astype(np.uint8)

        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth_img = depth_img.reshape((240, 320, 1))

        # depth_img = (cmap(depth)[..., :3] * 255).astype(np.uint8)[..., ::-1]  # RGB转BGR
        self.depth_img = depth_img
        # print(depth_img.shape) #(240, 320, 3)
        final_img,img_reward = self.get_final_image(depth_img,results)
        self.final_img = final_img
        self.img_reward = img_reward
        
        # #拼接显示
        if self.show_img:
            # tmp = (cmap(depth)[..., :3] * 255).astype(np.uint8)[..., ::-1]  # RGB转BGR
            depth_colormap = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
            # print(depth_colormap.shape)
            combined = cv2.hconcat([img_rgb, annotated_img,depth_colormap])
            cv2.imshow('Original | Detection | Depth | Final', combined)
            cv2.waitKey(1)

    def get_final_image(self,img_rgb, results):
        annotated_img = img_rgb.copy()
        # print(img_rgb.shape)
        mask = np.zeros_like(img_rgb)  # 创建一个与原图大小相同的黑色mask
        reward = 0  # 初始化奖励
        area = 0
        x_min, y_min, x_max, y_max = 0,0,0,0
        # 在原有代码基础上添加偏航控制逻辑
        if results and results[0].boxes:  # 确保有检测结果且有boxes
            boxes = results[0].boxes.xyxy.int().cpu()  # 获取检测框坐标，转换为整数
            img_height, img_width = img_rgb.shape[:2]  # 获取图像尺寸
            image_center_x = img_width // 2  # 图像水平中心坐标

            total_x_deviation = 0  # 累计水平偏差
            valid_boxes = 0  # 有效检测框计数器

            for box in boxes:
                x_min, y_min, x_max, y_max = box

                # 计算当前检测框的水平中心
                obj_center_x = (x_min + x_max) // 2

                # 计算水平偏差（像素单位）
                delta_x = obj_center_x - image_center_x
                # print("delta_x",delta_x)

                # 只考虑显著偏差（超过图像宽度5%）
                if abs(delta_x) > 0.05 * img_width:
                    total_x_deviation += delta_x
                    valid_boxes += 1

                # 原有面积计算逻辑
                area += (y_max - y_min) * (x_max - x_min)
                mask[y_min:y_max, x_min:x_max] = img_rgb[y_min:y_max, x_min:x_max]

                break

        # 计算奖励 #####################################################
        image_height = img_rgb.shape[0]  # 从输入图像获取高度
        image_width = img_rgb.shape[1]  # 从输入图像获取宽度
        x_center = (x_max+x_min)/2
        y_center = (y_max+y_min)/2
        # print("x_center",x_center)
        self.last_yaw_rate = 0.01 * float((x_center-image_width/2))

        annotated_img = mask  # 保持原有mask处理

        reward += area / (image_height * image_width) * 3 - 0.05 * abs(float(x_center - image_width / 2)) - 0.03 * abs(
            float((y_center - image_height / 2)))
        # center_difference = abs(((x_max-x_min)/2-image_width/2))+abs(((y_max-y_min)/2-image_height/2))
        # reward_center = (self.last_center_difference-center_difference)*0.1
        # self.last_center_difference = center_difference
        # print("reward_center", reward_center)
        
        if area>=self.last_image_area and area>10:
            reward+=0.8+(area/(image_height*image_width)-self.last_image_area/(image_height*image_width) )*4
        else:
            reward+=-0.8+(area/(image_height*image_width)-self.last_image_area/(image_height*image_width) )*4
            if self.previous_distance >1.5:
                reward += -2
        self.last_image_area = area
        self.img_state = np.array([x_min, y_min, x_max, y_max])
        # print("image reward:",reward)
        if self.show_img:
            show_img = annotated_img.copy()
            cv2.putText(show_img,
                        f"image reward: {reward:.2f}",  # 格式化reward保留两位小数
                        (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),  # 白色文字
                        2)
            # # 显示处理后的图像 ############################################
            cv2.imshow("Image State", cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        # resized_width = int(annotated_img.shape[1] * 0.5)
        # resized_height = int(annotated_img.shape[0] * 0.5)
        # resized_dimensions = (resized_width, resized_height)
        # annotated_img = cv2.resize(annotated_img, resized_dimensions, interpolation=cv2.INTER_AREA)
        # annotated_img = np.expand_dims(annotated_img, axis=-1)
        return annotated_img, reward  # 返回填充后的图像和计算出的奖励值
    def get_deep_image(self):
        responses = self.airsim_client.simGetImages([
            airsim.ImageRequest('front_center', airsim.ImageType.DepthVis, True, False)])
        img_depth_planar = np.array(responses[0].image_data_float).reshape(responses[0].height, responses[0].width)
        # 2. 距离100米以上的像素设为白色（此距离阈值可以根据自己的需求来更改）
        img_depth_vis = img_depth_planar / 0.3
        img_depth_vis[img_depth_vis > 1] = 1.
        # 3. 转换为整形
        img_depth_vis = (img_depth_vis * 255).astype(np.uint8)
        # cv2.imshow("Drone FPV", img_depth_vis)
        # cv2.waitKey(1)

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
        
        if(len(points)) < 3:
            return complete_points
        
        points = np.reshape(points, (int(points.shape[0]/3), 3))
        points_shape = np.shape(points);
        
        complete_points[:points_shape[0],:points_shape[1]] = points
        # print(complete_points.shape)
        # print("386 complete point",complete_points.shape) (3000, 3)
        return complete_points

    def get_god_state(self):
        # 获取无人机的状态
        self.next_gate = min(self.next_gate,len(self.gate_poses_ground_truth)-1)

        next_next_gate = min(self.next_gate+1,len(self.gate_poses_ground_truth)-1)

        linear_velocity = self.get_velocity()
        angular_velocity = self.get_angular_velocity_velocity()
        linear_acceleration = self.get_linear_acceleration()
        orientation = self.get_orientation()
        drone_pitch,drone_roll,drone_yaw,drone_angle = self.getPitchRollYaw()

        gate = self.gate_poses_ground_truth[self.next_gate]
        gate_pitch,gate_roll,gate_yaw,gate_angle = self.calculatePitchRollYaw(gate.orientation)

        drone_position = self.get_position()
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])

        nn_gate = self.gate_poses_ground_truth[next_next_gate]
        nn_position = np.array([nn_gate.position.x_val, nn_gate.position.y_val, nn_gate.position.z_val])

        # goal_direction = self.goal_direction(position, gate_position)
        relative_pos = gate_position - drone_position
        relative_yaw = gate_yaw -drone_yaw

        gate_oriention = np.array([gate.orientation.x_val, gate.orientation.y_val, gate.orientation.z_val,gate.orientation.w_val])

        relative_xyzw = quaternion_multiply(gate_oriention,orientation)

        qstate = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name)
        q = qstate.kinematics_estimated.orientation
        relative_pos = global_to_body_point(relative_pos,q)

        # state = np.concatenate((relative_pos,nn_position-gate_position,gate_oriention,np.array([drone_pitch,drone_roll,drone_yaw]),angular_velocity,linear_velocity))
        # state = np.concatenate((relative_pos,orientation,gate_oriention))
        # state = relative_pos
        # state = np.concatenate((relative_pos,np.array([relative_yaw])))
        img,img_reward,img_state = self.final_img,self.img_reward,self.img_state
        # state = np.concatenate((img_state,relative_pos))#训练时给予该信息
        state = np.concatenate((img_state,np.array([0,0,0])))#训练时给予该信息
        if self.observation_type=="raw_image":
            # state = relative_pos
            observation = {
                'vector': state,
                'image': self.camer_Image
            }
            return observation,img_reward
        # state = np.concatenate((img_state,relative_pos, orientation, gate_oriention))


        # state = np.concatenate((img_state,np.array([0,0,0])))#测试时置为0，效果依旧很好

        # state = relative_pos
        # print(f"relative pos:{relative_pos}")

        #保存cvae训练图片
        # if img_reward>0 and img_state[0]*img_state[1]!=0:
        #     cv2.imwrite(f'cvae/vae_test_imgs/{img_state[0]} {img_state[1]} {img_state[2]} {img_state[3]} {relative_pos[0]:.2f} {relative_pos[1]:.2f} {relative_pos[2]:.2f} '
        #               f'{orientation[0]:.2f} '
        #               f'{orientation[1]:.2f} {orientation[2]:.2f} {orientation[3]:.2f} {gate_oriention[0]:.2f} {gate_oriention[1]:.2f} '
        #               f'{gate_oriention[2]:.2f} {gate_oriention[3]:.2f}.jpg',img)

        #得到cvae条件向量
        conditions_tensor = torch.tensor(img_state,dtype=torch.float32).unsqueeze(0).to('cuda')
        # print("conditions",conditions_tensor.shape)
        # print(f"nimg:{img.shape}") #nimg:(240, 320, 1)

        #处理图片维度
        image_np_gray = img.squeeze()
        image_pil = Image.fromarray(image_np_gray) # 从 numpy array 创建 PIL Image
        image_tensor = self.cvae_transform(image_pil).unsqueeze(0)  # 添加 batch 维度
        image_tensor = image_tensor.to('cuda').float()

        # 进行前向传播，禁用梯度计算
        with torch.no_grad():
            if self.show_img:
                recon_imgs, mu, logvar = self.vae_model(image_tensor, conditions_tensor)
                cv2.imshow("recogimg", cv2.cvtColor(np.array(recon_imgs.cpu()).reshape(240,320,1), cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
            z = self.vae_model.get_z(image_tensor, conditions_tensor)
        observation = np.concatenate((state,z.flatten().cpu()))

        with open(f"trac1.txt","a") as f:
            f.write(f"position_x:{drone_position[0]} position_y:{drone_position[1]} position_z:{drone_position[2]}\n")
        # cv2.imshow("recogimg", cv2.cvtColor(np.array(recon_imgs.cpu()).reshape(240,320,1), cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        # from torchvision.utils import save_image
        # save_image(recon_imgs, f'cvae/vae_re_imgs/{cond_str}.png')
        # print("重构图像已保存为 reconstructed_sample.png")

        # observation = {
        #     'vector': state,
        #     'image': img
        # }
        return observation,img_reward

    # Resets a current race: moves players to start positions, timer and penalties reset
    def reset(self, seed=None,options=None):
        print("reset")
        self.airsim_client.simResetRace()
        self.stop_log_monitor_callback_thread()
        # self.stop_image_callback_thread()
        
        # if self.observation_type == "lidar":
        #     self.vis.destroy_window()

        self.previous_distance = 2
        self.last_image_area = 0
        self.next_gate = 0
        self.has_collission = False
        self.has_finished = False
        self.img_state = np.array([0,0,0,0])

        time.sleep(1)  # 等待3秒后开始下一个episode
        return self.start_race(),{}  # 开始新的比赛，获取初始状态
    
    def _do_action_angle_rate(self, action, step_length=1, duration=0.05):
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        angular_rates = state.angular_velocity
        quad_vel = state.linear_velocity  # 直接使用速度信息

        # 解析连续动作（假设 action 是四维数组）
        delta_roll = action[0] * step_length
        delta_pitch = action[1] * step_length
        delta_yaw = action[2] * step_length
        delta_vz = action[3] * step_length

        # 计算新的角速率
        new_roll_rate = angular_rates.x_val + delta_roll
        new_pitch_rate = angular_rates.y_val + delta_pitch
        new_yaw_rate = angular_rates.z_val + delta_yaw

        # 发送角速率控制指令（保持当前高度）
        self.airsim_client.moveByAngleRatesZAsync(
            roll_rate=new_roll_rate,
            pitch_rate=new_pitch_rate,
            yaw_rate=new_yaw_rate,
            z=state.position.z_val+delta_vz,
            duration=duration
        ).join()

    def _do_action_velocity1(self, action, step_length=1.0, duration=0.2):
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        current_vel = state.linear_velocity

        # 解析连续动作（4维：vx/vy/vz + yaw_rate）
        vx = action[0] * step_length
        vy = action[1] * step_length
        vz = action[2] * step_length

        gvx, gvy, gvz = body_to_global_velocity(vx, vy, vz, orientation=state.orientation)
        # delta_yaw = action[3] * 30  # 最大偏航速率30 deg/s

        # 生成新速度指令
        self.airsim_client.moveByVelocityAsync(gvx
                                               , gvy
                                               , gvz
                                               , duration=duration
                                               , drivetrain=airsim.DrivetrainType.ForwardOnly
                                               , yaw_mode=airsim.YawMode(is_rate=False)
                                               , vehicle_name=self.drone_name).join()

    def _do_action_velocity_discrete(self, action, step_length=1.0, duration=0.2):
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        vx = [-3,-1,1,3,5][action[0]]*step_length
        vy = [-2,-1,0,1,2][action[1]]*step_length
        vz = [-3,-1,0,1,3][action[2]]*step_length

        gvx, gvy, gvz = body_to_global_velocity(vx, vy, vz, orientation=state.orientation)
        # 生成新速度指令
        self.airsim_client.moveByVelocityAsync(gvx
                                               , gvy
                                               , gvz
                                               , duration=duration
                                               , drivetrain=airsim.DrivetrainType.ForwardOnly
                                               , yaw_mode=airsim.YawMode(is_rate=False)
                                               , vehicle_name=self.drone_name).join()

    def _do_action_velocity(self, action, step_length=1, duration=0.2):
        state = self.airsim_client.getMultirotorState().kinematics_estimated

        # 解析动作（前三维速度，第四维偏航速率）
        vx = action[0] * step_length
        vy = action[1] * step_length
        vz = action[2] * step_length
        yaw_rate = action[3] * 59.2757  # 弧度到度

        gvx, gvy, gvz = body_to_global_velocity(vx, vy, vz, state.orientation)
        # 使用最大自由度模式允许独立控制转向
        self.airsim_client.moveByVelocityAsync(
            gvx,gvy,gvz,
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,  # 修改drivetrain类型
            yaw_mode=airsim.YawMode(
                is_rate=True,  # 启用速率模式
                yaw_or_rate=yaw_rate  # 单位：度/秒
            ),
            vehicle_name=self.drone_name
        ).join()
        # self.airsim_client.moveByVelocityAsync(
        #     gvx,
        #     gvy,
        #     gvz,
        #     duration=duration,
        #     drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,  # 修改drivetrain类型
        #     yaw_mode=airsim.YawMode(
        #         is_rate=True,  # 启用速率模式
        #         yaw_or_rate=yaw_rate  # 单位：度/秒
        #     ),
        #     vehicle_name=self.drone_name
        # ).join()

    def _do_action_discrete(self, action_idx, step_length=1.0, duration=0.2):
        """离散动作执行函数

        Args:
            action_idx: 离散动作索引（0-12）
            step_length: 基础移动步长系数
            duration: 动作持续时间

        动作空间设计：
            0-7: 八方向水平移动（包含对角线）
            8-9: 垂直运动
            10-11: 偏航控制
            12: 悬停

        动作参数特点：
            1. 水平移动速度归一化
            2. 偏航速率限制为15度/秒
            3. 包含悬停动作
        """
        # 离散动作定义表 [vx_body, vy_body, vz_body, yaw_rate_deg]
        discrete_actions = [
            # 八方向水平移动（机体坐标系）
            (1, 0, 0, 0),  # 0: 前向
            (0.707, 0.707, 0, 0),  # 1: 右前45度
            (0, 1, 0, 0),  # 2: 右向
            (-0.707, 0.707, 0, 0),  # 3: 右后135度
            (-1, 0, 0, 0),  # 4: 后向
            (-0.707, -0.707, 0, 0),  # 5: 左后225度
            (0, -1, 0, 0),  # 6: 左向
            (0.707, -0.707, 0, 0),  # 7: 左前315度

            # 垂直运动
            (0, 0, 1, 0),  # 8: 上升
            (0, 0, -1, 0),  # 9: 下降

            # 偏航控制（速率模式）
            (0, 0, 0, 15),  # 10: 左转（15度/秒）
            (0, 0, 0, -15),  # 11: 右转（-15度/秒）
        ]
        # 参数校验
        if action_idx < 0 or action_idx >= len(discrete_actions):
            raise ValueError(f"Invalid action index: {action_idx}")

        # 解析动作参数
        vx_body, vy_body, vz_body, yaw_rate = discrete_actions[action_idx]

        # 获取当前状态
        state = self.airsim_client.getMultirotorState().kinematics_estimated
        current_orientation = state.orientation

        # 速度矢量归一化处理（仅水平方向）
        horizontal_speed = math.hypot(vx_body, vy_body)
        if horizontal_speed > 1e-6:
            scale = step_length / horizontal_speed
            vx_body *= scale
            vy_body *= scale

        # 转换到全局坐标系
        gvx, gvy, gvz = body_to_global_velocity(
            vx_body,
            vy_body,
            vz_body * step_length,  # 垂直方向单独缩放
            orientation=current_orientation
        )

        # 配置偏航模式
        if yaw_rate != 0:
            yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate*2)
        else:
            yaw_mode = airsim.YawMode(is_rate=False)
        # 执行控制指令
        self.airsim_client.moveByVelocityAsync(
            gvx*5,
            gvy*5,
            gvz*5,
            duration=duration,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,  # 允许横向移动
            yaw_mode=yaw_mode,
            vehicle_name=self.drone_name
        ).join()

    def _do_action_roll_pitch_yaw_throttle(self, action, duration=0.2):
        # 解包动作值
        roll, pitch, yaw, throttle = action

        # 发送滚转-俯仰-偏航-油门控制指令
        self.airsim_client.moveByRollPitchYawThrottleAsync(
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
            throttle=float(throttle),
            duration=duration
        ).join()
    def step(self, action):
        
        #print(action)
        #x = np.clip(action[0], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float) 
        #y = np.clip(action[1], -self.max_axis_velocity, self.max_axis_velocity).astype(np.float)
        #Read current state of the drone
        drone_state = self.airsim_client_odom.getMultirotorState()
        #Get quaternion to calculate rotation angle in Z axis (yaw)
        q = drone_state.kinematics_estimated.orientation
        angle = math.atan2(2.0 * (q.w_val * q.z_val + q.x_val * q.y_val) , 1.0 - 2.0 * (q.y_val * q.y_val + q.z_val * q.z_val))

        #Rotate dimensions using rotation matrix in 2D

        gate = self.gate_poses_ground_truth[self.next_gate]
        drone_position = self.get_position()
        gate_position = np.array([gate.position.x_val, gate.position.y_val, gate.position.z_val])
        gate_position += np.random.rand(3)
        direction_vector = gate_position - drone_position
        direction_vector /= np.linalg.norm(direction_vector)
        direction_vector *= 5
        state = self.airsim_client.getMultirotorState(vehicle_name=self.drone_name)
        q = state.kinematics_estimated.orientation
        relative_pos_body = global_to_body_point(direction_vector,q)
        # print(direction_vector)
        # self.airsim_client.moveByVelocityZAsync(x
        #                                         , y
        #                                         , z = gate.position.z_val-0.5
        #                                         , duration = 0.1
        #                                         , drivetrain= airsim.DrivetrainType.ForwardOnly
        #                                         , yaw_mode= airsim.YawMode(is_rate=False)
        #                                         , vehicle_name=self.drone_name)
        # self.airsim_client.moveByVelocityAsync(x
        #                                         , y
        #                                         , z
        #                                         , duration = 0.1
        #                                         , drivetrain= airsim.DrivetrainType.ForwardOnly
        #                                         , yaw_mode= airsim.YawMode(is_rate=False)
        #                                         , vehicle_name=self.drone_name)
        # move_by_path_3d(self.airsim_client,[airsim.Vector3r(gate_position[0],gate_position[1],gate_position[2])])
        # self._do_action_angle_rate(action,step_length=0.5, duration=0.3)

        # if self.last_image_area>0:
        #     self._do_action_velocity(action,step_length=1,duration=0.2)
        # else:
        #     self._do_action_velocity1(action, step_length=1, duration=0.2)


        self._do_action_velocity(action, step_length=1, duration=0.03)
        # self._do_action_velocity_discrete(action, step_length=1, duration=0.05)


        # self._do_action_angle_rate(action, duration=0.05)

        # self._do_action_roll_pitch_yaw_throttle(action, duration=0.1)

        # self._do_action_discrete(action,1,0.2)
        done = False
        truncated = False
        (reward, done,truncated) = self.calculate_reward(action,verbose=True)

        # action_difference = np.linalg.norm(action - self.last_action)

        # 计算惩罚值，偏转方向
        action_penalty = (-0.3*abs(action[1]) + -0.05*abs(action[2]))
        self.last_action = action

        # print(reward)
        info = {"next_gates": self.next_gate}
        state,img_reward = self.get_observation()
        if reward>0:
            reward += float(img_reward)*0.1
        print("img reward: ",img_reward*0.1)

        # reward += action_penalty
        # print("action:",action)
        print()
        # print(f"image_reward:{img_reward} | all reward:{reward}")
        # print("Image reward:",img_reward)

        return state, reward, done,truncated,info

    def get_ground_truth_gate_facing(self):
        self.gate_facing = [self.get_gate_facing_vector_from_quaternion(gatepose.orientation)
                            for gatepose in self.gate_poses_ground_truth]
            

    