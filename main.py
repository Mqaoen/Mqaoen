# -*- coding: utf-8 -*-
# @Time    : 2024/9/12 16:01
# @Author  : Xiaowen Qian
# @Site    : 
# @File    : main.py
# @Software: PyCharm 
# @Comment :

import cv2
from ultralytics import YOLO

# 加载 YOLOv8 模型 (使用 'yolov8n.pt' 作为轻量模型)
model = YOLO('yolov8n.pt')  # 可以换成 'yolov8s.pt', 'yolov8m.pt' 等

# 打开摄像头
cap = cv2.VideoCapture(0)  # 参数 0 表示默认摄像头

if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置摄像头分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)  # 设置宽度，例如 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # 设置高度，例如 720

# 实时处理视频流
while True:
    ret, frame = cap.read()

    if not ret:
        print("无法读取帧")
        break

    # 使用 YOLO 模型进行检测
    results = model(frame)

    # 遍历每个检测结果，绘制边框和标签
    for result in results:
        for box in result.boxes:
            # 获取边框坐标、类别和置信度
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 边框坐标
            conf = box.conf[0]  # 置信度
            cls = int(box.cls[0])  # 类别索引

            # 绘制边框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{model.names[cls]} {conf:.2f}"  # 类别标签和置信度
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # 显示检测后的帧，并设置窗口大小为摄像头分辨率
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()


