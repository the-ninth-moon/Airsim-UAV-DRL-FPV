本科毕设

基于单目相机感知的，使用强化学习控制的无人机穿越多个门框任务

在自定义的环境中，门框的命名需要按照Gate0、Gate1……的顺序才能被程序识别

参考[https://zhuanlan.zhihu.com/p/512327050](https://zhuanlan.zhihu.com/p/512327050)实现了一份PPO算法，可以使用main.py来训练。
实现了gym接口，所以也可以使用stablebaseline3的PPO算法，在stbmain_3laststate.py中进行了3帧的帧堆叠并使用PPO进行训练。

在UE4环境中训练

![媒体1 00_00_03--00_00_23](https://github.com/user-attachments/assets/ebff97e6-cef3-4e95-bd9b-bac7cdb2db1e)

直接迁移到airsim挑战赛中

![lab 00_00_01--00_00_21](https://github.com/user-attachments/assets/998618a2-5d02-44f4-bfee-f0b5e202f6ab)

可视化：

![2](https://github.com/user-attachments/assets/7d081b18-c49c-4892-8dd5-ab108a3d7641)


整体框架：

![image](https://github.com/user-attachments/assets/ac86788d-82bc-45ce-aa80-0f893265974a)
