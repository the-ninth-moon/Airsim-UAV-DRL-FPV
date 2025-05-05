import airsimdroneracinglab as airsim
import numpy as np
import math

def get_state(client):
    # 获取无人机状态
    DIG = 6
    State = client.getMultirotorState()
    kinematics = State.kinematics_estimated
    state = {
        "timestamp": str(State.timestamp),
        "position": [round(ele, DIG) for i, ele in
                     enumerate(kinematics.position.to_numpy_array().tolist())],
        "orientation": [round(i, DIG) for i in airsim.to_eularian_angles(kinematics.orientation)],
        "linear_velocity": [round(i, DIG) for i in kinematics.linear_velocity.to_numpy_array().tolist()],
        "linear_acceleration": [round(i, DIG) for i in kinematics.linear_acceleration.to_numpy_array().tolist()],
        "angular_velocity": [round(i, DIG) for i in kinematics.angular_velocity.to_numpy_array().tolist()],
        "angular_acceleration": [round(i, DIG) for i in kinematics.angular_acceleration.to_numpy_array().tolist()]
    }
    return state


def move_by_acceleration_horizontal(client, ax_cmd, ay_cmd, az_cmd, z_cmd, duration=0.04):
    # 读取自身yaw角度
    state = get_state(client)
    angles = state['orientation']
    yaw_my = angles[2]
    g = 98  # 重力加速度
    sin_yaw = math.sin(yaw_my)
    cos_yaw = math.cos(yaw_my)
    A_psi = np.array([[cos_yaw, sin_yaw], [-sin_yaw, cos_yaw]])
    A_psi_inverse = np.linalg.inv(A_psi)
    angle_h_cmd = 1 / (g - az_cmd) * np.dot(A_psi_inverse, np.array([[ax_cmd], [ay_cmd]]))
    theta = math.atan(angle_h_cmd[0, 0])
    phi = math.atan(angle_h_cmd[1, 0] * math.cos(theta))
    # client.moveToZAsync(z_cmd, vz).join()
    client.moveByRollPitchYawZAsync(phi, theta, 0, z_cmd, duration)


def move_by_path_3d(client, Path, K0=1.5, K1=4, K2=0.6, dt=0.04, a0=1, delta=0.7):
    def distance(A, B):
        return math.sqrt((A[0] - B[0]) ** 2 +
                         (A[1] - B[1]) ** 2 +
                         (A[2] - B[2]) ** 2)
    state = get_state(client)
    P = state['position']
    V = state['linear_velocity']
    Wb = P
    Wb_m = np.matrix(Wb).T
    P_m = np.matrix(P).T
    V_m = np.matrix(V).T
    I3 = np.matrix([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])
    for i in range(len(Path)):
        Wa = Wb
        Wb = [Path[i].x_val, Path[i].y_val, Path[i].z_val]
        Wa_m = Wb_m
        Wb_m = np.matrix(Wb).T
        A = I3 - (Wa_m - Wb_m).dot((Wa_m - Wb_m).T) / (distance(Wa_m, Wb_m) ** 2)
        Pt = P_m - Wb_m
        e = np.linalg.norm(A.dot(Pt))
        d = np.linalg.norm(Pt - A.dot(Pt))
        while d >= delta or \
                (i == len(Path) - 1
                 and ((P[0] - Wb[0]) * (Wb[0] - Wa[0]) < 0
                      or (P[1] - Wb[1]) * (Wb[1] - Wa[1]) < 0
                      or (P[2] - Wb[2]) * (Wb[2] - Wa[2]) < 0)):
            Pt = P_m - Wb_m
            U1 = K0 * Pt + K1 * A.dot(Pt)
            if np.linalg.norm(U1, ord=np.inf) > a0:
                U1 = U1 * a0 / np.linalg.norm(U1, ord=np.inf)
            U = -(U1 + V_m) / K2
            U_cmd = np.array(U)[:, 0]
            z_cmd = P[2] + (V[2] + U_cmd[2] * dt) * dt
            move_by_acceleration_horizontal(client, U_cmd[0], U_cmd[1], U_cmd[2], z_cmd, dt)
            e = np.linalg.norm(A.dot(Pt))
            d = np.linalg.norm(Pt - A.dot(Pt))
            # 画图
            plot_p1 = [airsim.Vector3r(P[0], P[1], P[2])]
            state = get_state(client)
            P = state['position']
            V = state['linear_velocity']
            P_m = np.matrix(P).T
            V_m = np.matrix(V).T
            plot_p2 = [airsim.Vector3r(P[0], P[1], P[2])]
            # client.simPlotArrows(plot_p1, plot_p2, arrow_size=8.0, color_rgba=[0.0, 0.0, 1.0, 1.0])
            # client.simPlotLineStrip(plot_p1 + plot_p2, color_rgba=[1.0, 0.0, 0.0, 1.0], is_persistent=True)


