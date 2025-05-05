# ------------ Color and Edge Based Snake Gate Detection -------------#
# ------------------- For Autonomoous Drone Racing -------------------#
# --------------------- By Weronika Dziarnowska ----------------------#
# Developed for the course "Autonomous Flight of Micro Air Vehicles" #
# ---------------- At Delft University of Technology -----------------#

# To run the code on an image, scroll all the way down and enter the
# image name and path.

import numpy as np
import cv2

def process_image(image, kernel_size=15, erosion_iters=2, dilation_iters=3):
    """
    处理图像：生成门区域的mask，执行背景模糊、前景锐化等操作，增强门区域的可见性。

    参数:
    - image: 输入图像（BGR格式）
    - kernel_size: 高斯模糊核大小
    - erosion_iters: 腐蚀操作的迭代次数
    - dilation_iters: 膨胀操作的迭代次数

    返回:
    - 处理后的图像
    """

    # 1. 颜色空间转换：将图像从BGR转换为HSV
    im_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 2. 生成门的掩膜（mask），通过绿色和白色的颜色范围
    lower_green = np.array([20, 50, 50])  # H: 35-85, S: 50-255, V: 50-255
    upper_green = np.array([100, 255, 255])

    # 白色的 HSV 范围
    lower_white = np.array([0, 0, 180])  # H: 0-179, S: 0-30, V: 180-220
    upper_white = np.array([179, 30, 255])

    # 提取绿色区域
    green_mask = cv2.inRange(im_hsv, lower_green, upper_green)
    # 提取白色区域
    white_mask = cv2.inRange(im_hsv, lower_white, upper_white)

    # 合并绿色和白色区域，生成最终的掩膜
    combined_mask = cv2.bitwise_or(green_mask, white_mask)

    # 3. 背景模糊处理
    blurred_im = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # 4. 前景锐化：对图像进行锐化处理
    kernel_sharpen = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])  # 锐化核
    sharpened_im = cv2.filter2D(blurred_im, -1, kernel_sharpen)

    # 5. 使用腐蚀去噪声
    kernel = np.ones((2, 2), np.uint8)
    eroded_mask = cv2.erode(combined_mask, kernel, iterations=erosion_iters)

    # 6. 使用膨胀填补门区域的空隙
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=dilation_iters)

    # 7. 强化前景（门区域）：通过降低背景亮度来增强门区域的突出性
    final_im = sharpened_im.copy()
    final_im[dilated_mask == 0] = final_im[dilated_mask == 0] * 0.3  # 背景减亮

    return final_im




def post_process_mask(mask):
    # 定义一个3x3的结构元素
    kernel = np.ones((3, 3), np.uint8)
    # 先腐蚀再膨胀，可以去噪声并填补空隙
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask


def color_filter(im):
    # 绿色的 HSV 范围
    lower_green = np.array([20, 50, 50])  # H: 35-85, S: 50-255, V: 50-255
    upper_green = np.array([100, 255, 255])

    # 白色的 HSV 范围
    lower_white = np.array([0, 0, 180])  # H: 0-179, S: 0-30, V: 180-220
    upper_white = np.array([179, 30, 255])

    # 进行颜色检测
    filtered_green = cv2.inRange(im, lower_green, upper_green)
    filtered_white = cv2.inRange(im, lower_white, upper_white)

    # 合并两个颜色检测的结果
    filtered = cv2.bitwise_or(filtered_green, filtered_white)

    filtered = post_process_mask(filtered)

    return filtered


# 使用Canny边缘检测算法检测图像边缘
def edge_detector(im):
    # 使用cv2.Canny()函数进行边缘检测，300是低阈值，800是高阈值
    # 该函数返回一个二值图像，边缘像素为255，其他像素为0
    return cv2.Canny(im, 400, 500)


# 分类图像窗口是否为门框部分（根据颜色和边缘检测结果）
def detector(window, window_edges):
    # 对当前窗口中的颜色进行过滤，得到蓝色区域
    window_color = color_filter(window)
    # 已经计算得到的边缘信息直接传入
    window_edges = window_edges
    # 计算窗口中蓝色像素的数量
    color_nr = (window_color > 0).sum()
    # 计算窗口中边缘像素的数量
    edge_nr = (window_edges > 0).sum()
    # 如果蓝色区域和边缘区域的像素数量都超过某个阈值，则认为这是一个门框部分
    # 阈值为图像窗口大小的30%（即窗口内像素数量的30%）
    if (color_nr >= 0.1 * len(window) ** 2) and (edge_nr >= 0.1 * len(window)):
        detected = 1  # 检测到门框
    else:
        detected = 0  # 未检测到门框

    return detected

# Detect gate elements in the entire image.
# This function is designed to apply combined color and edge detection
# to find potential gate elements throughout the entire image.
# It outputs a mask indicating where gates are detected in the image.
# The function is not used in the snake search but was developed for tuning.
def color_edge(im, im_edges, kernel):
    size = len(im)  # 获取图像的大小（假设图像是正方形）

    # Kernel大小应为奇数，所以计算“半”值来进行索引
    half = int((kernel - 1) / 2)

    # 初始化从底部开始的y坐标
    y = size - half

    # 创建一个与图像大小相同的全零矩阵，用于存储检测到的门框位置
    detected_gate = np.zeros((size, size))

    # 在图像的垂直方向上遍历
    while (y - half >= 0):
        x = half  # 从水平最左边开始
        # 在水平方向上遍历图像
        while (x + half <= size):
            # 获取当前窗口的颜色图像和边缘图像
            window = im[y - half:y + half, x - half:x + half]
            window_edges = im_edges[y - half:y + half, x - half:x + half]

            # 使用detector函数判断当前窗口是否包含门框
            detected = detector(window, window_edges)

            # 如果检测到门框，标记为1
            if detected == 1:
                detected_gate[y - half:y + half, x - half:x + half] = 1

            # 向右移动
            x = x + 1

        # 向上移动
        y = y - 1

    return detected_gate


# Snake search function for detecting the gate.
# The main function searches for the left vertical bar and two horizontal bars in the gate.
def snake_gate_detection(im, im_edges, kernel, sigma):
    size = len(im)  # 获取图像的大小（假设图像是正方形）

    # Kernel大小应为奇数，所以计算“半”值来进行索引
    half = int((kernel - 1) / 2)

    # 初始化y坐标从底部开始
    y = size - half

    # 预先定义found标志用于判断是否成功检测到门框
    found = 0
    done = 0  # 标志位，标记搜索是否完成

    # 遍历图像的垂直方向
    while (done == 0) and (y - half >= 0):
        x = half  # 从水平最左边开始
        # 遍历图像的水平方向
        while (done == 0) and (x + half <= size):
            # 获取当前窗口的边缘检测结果
            window_edges = im_edges[y - half:y + half, x - half:x + half]

            # 如果在当前窗口检测到边缘，进一步判断是否为门框
            if np.any(window_edges > 0):
                # 获取当前窗口的图像
                window = im[y - half:y + half, x - half:x + half]

                # 使用detector函数检测当前窗口是否为门框
                detected = detector(window, window_edges)
                if detected == 1:
                    # 如果检测到门框，记录下P1为左下角的位置
                    P1y = y
                    P1x = x

                    # 向上搜索P2（左上角位置）
                    [P2y, P2x] = search_up(y, x, im, im_edges, kernel)

                    # 计算P1与P2之间的距离，若大于sigma，则认为找到有效的左竖条
                    if np.linalg.norm([P1y - P2y, P1x - P2x]) >= sigma:
                        # 向右搜索P3和P4（右下角和右上角位置）
                        [P3y, P3x] = search_right(P1y, P1x, im, im_edges, kernel)
                        [P4y, P4x] = search_right(P2y, P2x, im, im_edges, kernel)

                        # 如果任何一条横杆足够长，则接受检测结果
                        if (np.linalg.norm([P1y - P3y, P1x - P3x]) >= sigma) or (
                                np.linalg.norm([P2y - P4y, P2x - P4x]) >= sigma):
                            done = 1  # 完成检测
                            found = 1  # 成功找到门框
            # 如果还没有找到门框，继续向右移动
            x = x + half

        # 向上移动
        y = y - half

    # 如果找到门框，进一步优化角点位置并返回
    if found == 1:
        points = np.array([[P1y, P1x], [P2y, P2x], [P3y, P3x], [P4y, P4x]])
        points = refine_corners(points, sigma)  # 优化角点
        return points, found

    # 如果未找到门框，返回found为0
    else:
        return found, found


# Function to search up. It works according to the original algorithm but it
# also accepts x locations 2 pixels away, not just 1 pixel away.
import matplotlib.pyplot as plt


# Function to search upwards from a given point in the image to find the top of the gate.
# This function allows the search to move up 1 or 2 pixels left or right at each step.
def search_up(y, x, im, im_edges, kernel):
    done = 0  # 标记是否完成搜索
    size = len(im)  # 获取图像的尺寸
    half = int((kernel - 1) / 2)  # 计算窗口的一半大小

    # 一直搜索直到找到门框的顶部或超出图像边界
    while (done == 0) and (y - 1 - half >= 0) and (x - 2 - half >= 0) and (x + 2 + half <= size):
        # 检查当前窗口是否包含门框，如果是，向上移动
        if detector(im[y - 1 - half:y - 1 + half, x - half:x + half],
                    im_edges[y - 1 - half:y - 1 + half, x - half:x + half]) == 1:
            y = y - 1
        # 允许窗口左移一格
        elif detector(im[y - 1 - half:y - 1 + half, x - 1 - half:x - 1 + half],
                      im_edges[y - 1 - half:y - 1 + half, x - 1 - half:x - 1 + half]) == 1:
            y = y - 1
            x = x - 1
        # 允许窗口右移一格
        elif detector(im[y - 1 - half:y - 1 + half, x + 1 - half:x + 1 + half],
                      im_edges[y - 1 - half:y - 1 + half, x + 1 - half:x + 1 + half]) == 1:
            y = y - 1
            x = x + 1
        # 允许窗口左移两格
        elif detector(im[y - 1 - half:y - 1 + half, x - 2 - half:x - 2 + half],
                      im_edges[y - 1 - half:y - 1 + half, x - 2 - half:x - 2 + half]) == 1:
            y = y - 1
            x = x - 2
        # 允许窗口右移两格
        elif detector(im[y - 1 - half:y - 1 + half, x + 2 - half:x + 2 + half],
                      im_edges[y - 1 - half:y - 1 + half, x + 2 - half:x + 2 + half]) == 1:
            y = y - 1
            x = x + 2
        else:
            done = 1  # 如果未找到门框，则结束搜索

    return y, x  # 返回更新后的坐标


# Function to search to the right. It works similarly to search_up, allowing
# for adjustments of up to 2 pixels vertically.
def search_right(y, x, im, im_edges, kernel):
    done = 0  # 标记是否完成搜索
    size = len(im)  # 获取图像的尺寸
    half = int((kernel - 1) / 2)  # 计算窗口的一半大小

    # 一直搜索直到找到门框的右边或超出图像边界
    while done == 0 and (y - 1 - half >= 0) and (y + 1 + half <= size) and (x + 1 + half <= size):
        # 检查当前窗口右侧是否包含门框
        if detector(im[y - half:y + half, x + 1 - half:x + 1 + half],
                    im_edges[y - half:y + half, x + 1 - half:x + 1 + half]) == 1:
            x = x + 1
        # 允许窗口向下移动一格
        elif detector(im[y + 1 - half:y + 1 + half, x + 1 - half:x + 1 + half],
                      im_edges[y + 1 - half:y + 1 + half, x + 1 - half:x + 1 + half]) == 1:
            y = y + 1
            x = x + 1
        # 允许窗口向上移动一格
        elif detector(im[y - 1 - half:y - 1 + half, x + 1 - half:x + 1 + half],
                      im_edges[y - 1 - half:y - 1 + half, x + 1 - half:x + 1 + half]) == 1:
            y = y - 1
            x = x + 1
        # 允许窗口向下移动两格
        elif detector(im[y + 2 - half:y + 2 + half, x + 1 - half:x + 1 + half],
                      im_edges[y + 2 - half:y + 2 + half, x + 1 - half:x + 1 + half]) == 1:
            y = y + 2
            x = x + 1
        # 允许窗口向上移动两格
        elif detector(im[y - 2 - half:y - 2 + half, x + 1 - half:x + 1 + half],
                      im_edges[y - 2 - half:y - 2 + half, x + 1 - half:x + 1 + half]) == 1:
            y = y - 2
            x = x + 1
        else:
            done = 1  # 如果未找到门框，则结束搜索

    return y, x  # 返回更新后的坐标


# Function to refine the corners if one of the horizontal bars is too short.
# If the detected bars are shorter than a threshold (sigma), the function
# adjusts the corners to make the bars longer using vectors from the other bars.
def refine_corners(points, sigma):
    p = points  # 接收四个角点的坐标

    # 如果下横杆太短，使用上横杆的向量来调整
    if np.linalg.norm(p[2] - p[0]) < sigma:
        p[2] = p[0] + (p[3] - p[1])
    # 如果上横杆太短，使用下横杆的向量来调整
    elif np.linalg.norm(p[3] - p[1]) < sigma:
        p[3] = p[1] + (p[2] - p[0])

    return points  # 返回调整后的四个角点


# Function to process the image, detect the gate, and visualize the results.
def process_and_show_image(image):
    # 读取图像
    im = cv2.imread(image)

    # im = process_image(im)

    # 将图像从BGR转换为HSV色彩空间
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # 使用颜色过滤检测门（蓝色区域），用于可视化
    im_color = color_filter(im_hsv)

    # 检测图像的边缘
    im_edges = edge_detector(im_hsv)
    # im_edges = post_process_mask()

    # 基于颜色和边缘检测创建门框掩码，仅用于可视化
    im_gate = color_edge(im_hsv, im_edges, 15)

    # 执行蛇形搜索以检测门框
    [corners, found] = snake_gate_detection(im_hsv, im_edges, 20, 90)

    # 绘制并显示结果
    fig, axs = plt.subplots(1, 3, figsize=(35, 10))
    axs[0].imshow(im_color, aspect="auto")
    axs[1].imshow(im_edges, aspect="auto")
    axs[2].imshow(cv2.cvtColor(im_hsv, cv2.COLOR_HSV2RGB), aspect="auto")

    if found == 1:
        print("FOUND")  # 打印找到的信息
        # 在图像上绘制检测到的门框角点
        axs[2].scatter(corners[0][1], corners[0][0], s=500, c='red')
        axs[2].scatter(corners[1][1], corners[1][0], s=500, c='red')
        axs[2].scatter(corners[2][1], corners[2][0], s=500, c='red')
        axs[2].scatter(corners[3][1], corners[3][0], s=500, c='red')

    # 如果需要将图像保存到文件夹，可以取消注释下面的两行
    # number = image[11:-1]
    # fig.savefig("Plots_ColorTune/snake_" + number + "g")

    plt.show()

    return corners  # 返回角点


# ------------------ Enter your image name and path here ------------------#
# -- Uncomment and use the commands for multiple or only one image below --#

# Detect gates in all images from the folder.
# for image in glob.glob("Images/*.png"):
#     process_and_show_image(image)

# Or run the code just for one image.
image = r"D:\GraduationDesign\airsim-drl-reinforcement-learning\vae_train_imgs\4.jpg"
process_and_show_image(image)
# image = cv2.imread(image)
# image = process_image(image)
# cv2.imshow("1",image)
# cv2.waitKey(0)
