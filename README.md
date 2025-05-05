基于单目相机感知的，使用强化学习控制的无人机穿越多个门框任务

在自定义的环境中，门框的命名需要按照Gate0、Gate1……的顺序才能被程序识别

参考https://zhuanlan.zhihu.com/p/512327050实现了一份PPO算法，但是效果并不是特别好。可以使用main.py来训练。
实现了gym接口，可以使用stablebaseline3的PPO算法，stbmain_3laststate.py。其中进行了3帧的帧堆叠。

