import time
import cv2
import joblib
import numpy as np
from PIL import ImageGrab
from pynput import keyboard
from pynput.keyboard import Key

time.sleep(3)
# 0、创建键盘
kb = keyboard.Controller()
# 1、加载模型
model = joblib.load('auto_play.m')
while True:
    # 2、准备数据    截图 -> resize (500, 350) -> reshape 1 dim
    ImageGrab.grab().resize((960, 540)).save('current.jpg')  # 保存当前屏幕截屏
    x = cv2.imread('current.jpg', 0).reshape(-1)
    x = [x]
    # 3、预测
    pred = model.predict(x)
    print(pred)
    if pred[0] == 0:
        kb.press(Key.space)
