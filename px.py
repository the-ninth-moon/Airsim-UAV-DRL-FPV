import math

from pymavlink import mavutil



# while True:
# 	try:
# 		msg = master.recv_match(type='ATTITUDE', blocking=True)
# 		if not msg:
# 			raise ValueError()
# 		print(msg.to_dict())
# 	except KeyboardInterrupt:
# 		print('Key bordInterrupt! exit')
# 		break
import time


# 连接到飞控
def connect_to_vehicle(port='/dev/ttyAMA0', baud=57600):
    master = mavutil.mavlink_connection('udp:0.0.0.0:{}'.format(14550))  # port 是端口号
    print("Heartbeat from system (system %u component %u)" % (
        master.target_system, master.target_system))
    return master


# 等待心跳消息
def wait_for_heartbeat(master):
    print("Waiting for heartbeat...")
    master.wait_heartbeat()
    print("Heartbeat received from system (system %u component %u)" %
          (master.target_system, master.target_component))


# 解锁无人机
def arm_drone(master):
    print("Arming motors")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
        0,
        1, 0, 0, 0, 0, 0, 0)
    # 确认无人机已解锁
    master.motors_armed_wait()
    print("Motors armed")


# 起飞至指定高度
def takeoff(master, altitude):
    print("Taking off to %d meters" % altitude)
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
        0,
        0, 0, 0, 0, 0, 0, altitude)
    # 等待无人机达到目标高度
    while True:
        alt = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True).alt / 1000.0
        if alt >= altitude * 0.95:
            print("Reached target altitude: %d meters" % altitude)
            break
        time.sleep(1)


# 设置目标位置
def set_target_position(master, latitude, longitude, altitude):
    master.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(
        10,  # time_boot_ms (not used)
        master.target_system,  # target system
        master.target_component,  # target component
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,  # frame
        0b0000111111111000,  # type_mask (only positions enabled)
        int(latitude * 1e7),  # lat_int
        int(longitude * 1e7),  # lon_int
        altitude,  # alt
        0, 0, 0,  # vx, vy, vz
        0, 0, 0,  # afx, afy, afz
        0, 0))  # yaw, yaw_rate


# 绕圈飞行
def circle_flight(connection, radius, center_latitude, center_longitude, num_circles=3):
    for _ in range(num_circles):
        for angle in range(0, 360, 1):  # 每度发送一个命令
            x = center_latitude + (radius * math.cos(math.radians(angle)) / 111319.9)  # 计算经度增量
            y = center_longitude + (radius * math.sin(math.radians(angle)) / 111319.9)  # 计算纬度增量
            connection.mav.send(mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                10,  # 时间间隔（毫秒）
                connection.target_system,
                connection.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,  # 使用位置和速度
                int(x * 1e7),  # 目标纬度
                int(y * 1e7),  # 目标经度
                20,  # 目标高度
                0, 0, 0,  # x, y, z 速度
                0, 0, 0,  # x, y, z 加速度
                0, 0))  # 偏航角度和偏航速率
            time.sleep(0.1)  # 等待一段时间发送下一个位置


# 暂停一段时间
def pause(seconds):
    print("Pausing for %d seconds" % seconds)
    time.sleep(seconds)


# 返回起飞点
def return_to_launch(master):
    print("Returning to launch")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
        0, 0, 0, 0, 0, 0, 0, 0)
    # 等待返回起飞点
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg.relative_alt <= 1.0:
            print("Returned to launch")
            break
        time.sleep(1)


# 降落
def land_drone(master):
    print("Landing")
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        mavutil.mavlink.MAV_CMD_NAV_LAND,
        0, 0, 0, 0, 0, 0, 0, 0)
    # 等待降落完成
    while True:
        msg = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
        if msg.relative_alt <= 0.1:
            print("Landed")
            break
        time.sleep(1)


# 主函数
def main():
    # 连接飞控
    master = connect_to_vehicle()
    wait_for_heartbeat(master)

    # 解锁无人机
    arm_drone(master)

    # 起飞至20米
    takeoff(master, 20)

    # 获取起飞点坐标
    home_location = master.recv_match(type='GLOBAL_POSITION_INT', blocking=True)

    # 飞行矩形路径
    points = [
        (home_location.lat + 20000, home_location.lon),  # 20米北
        (home_location.lat + 20000, home_location.lon + 20000),  # 20米东
        (home_location.lat, home_location.lon + 20000),  # 20米南
        (home_location.lat, home_location.lon)  # 回到起飞点
    ]

    for lat, lon in points:
        set_target_position(master, lat / 1e7, lon / 1e7, 20)
        pause(5)

    # 返回起飞点
    return_to_launch(master)

    # 执行绕圈飞行
    circle_flight(master, 5, home_location.lat, home_location.lon)

    # 返回起飞点
    return_to_launch(master)

    # 降落
    land_drone(master)


if __name__ == "__main__":
    main()