import glob
import os
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import freeze_support

import cv2
import numpy as np
from keras.models import load_model


model_path = r'c:\prj\nalogru-captcha\Models\model.h5'
DIR = r'c:\prj\png-captcha'

channels = 1
pic_h = 100
pic_w = 200
len_captcha = 6
img_shape = (pic_h, pic_w, channels)
model = load_model(model_path)
symbols = ["0", "5", "9", "7", "6", "8", "1", "4", "2", "3"]
len_symbols = 10


def predict(captcha):
    captcha = np.reshape(captcha, (1, pic_h, pic_w, channels))
    result = model.predict(captcha)
    label = ''.join([symbols[np.argmax(i)] for i in result])
    return label


def solve(file):
    """
    @param file: полный путь до файла капчи
    @return: распознанное значение
    """
    captcha_cv2 = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    captcha_cv2 = captcha_cv2 / 255.0
    expectation = os.path.basename(file).split(".")[0]
    pred = predict(captcha_cv2)
    return pred


file = r'c:\prj\nalog-captcha\027718.jpg'
pred = solve(file)
print(f'Распознание файла: {file}', pred)


# check accuracy
# good = 0
# bad = 0
# total = 0
# for f in os.listdir(DIR):
#     f_path = os.path.join(DIR, f)
#     captcha_cv2 = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
#     captcha_cv2 = captcha_cv2 / 255.0
#     expectation = f.split(".")[0]
#     pred = predict(captcha_cv2)
#     if pred == expectation:
#         good += 1
#     else:
#         bad += 1
#     print(f'Распознание файла: {f_path}', pred)
#     break
# print(f'Total: {good + bad}')
# print(f'Good: {good}')
# print(f'Bad: {bad}')

