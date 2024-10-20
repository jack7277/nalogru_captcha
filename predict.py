import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import freeze_support

import cv2
import numpy as np
from keras.models import load_model


model_path = r'c:\prj\nalogru-captcha\Models\model.h5'
unk_path = r'c:\prj\nalogru-unk-pics'
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


# file = r'c:\prj\nalog-captcha\027718.jpg'
file = r'c:\prj\nalogru-unk-pics\35181.jpg'
file = r'c:\prj\nalogru-unk-pics\62316.jpg'
file = r'c:\prj\nalogru-unk-pics\64858.jpg'  # 642858
file = r'c:\prj\nalogru-unk-pics\73469C3C5FEA8F16B59DE8AF8C2B85D98F7B7A06D7D6E0796E12E613B5E09018.jpg'
# pred = solve(file)
# print(f'Распознание файла: {file}', pred)
# sys.exit()

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

# recognize
# unk_files = glob.glob(os.path.join(unk_path, '*.jpg'))
# for unk_file in unk_files:
#     print(unk_file)
#     pred = solve(unk_file)
#     new_full_name = os.path.join(unk_path, f'{pred}.jpg')
#     try:
#         os.rename(unk_file, new_full_name)
#     except Exception as e:
#         print(str(e))

