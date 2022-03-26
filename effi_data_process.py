import os
import shutil

import tqdm
from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load

folder_path_train_image = 'train'
folder_path_val_image = 'val'
folder_path_test_image = 'test'

'''
if not os.path.exists(folder_path_train):
    os.makedirs(folder_path_train)
if not os.path.exists(folder_path_val):
    os.makedirs(folder_path_val)
if not os.path.exists(folder_path_test):
    os.makedirs(folder_path_test)
if not os.path.exists(folder_path_train_image):
    os.makedirs(folder_path_train_image)
if not os.path.exists(folder_path_val_image):
    os.makedirs(folder_path_val_image)
if not os.path.exists(folder_path_test_image):
    os.makedirs(folder_path_test_image)
'''
os.makedirs('train/cm_front', exist_ok = True)
os.makedirs('train/cc_2_front', exist_ok = True)
os.makedirs('train/cm_back', exist_ok = True)
os.makedirs('train/cc_back', exist_ok = True)
os.makedirs('train/dl_front', exist_ok = True)
os.makedirs('train/cc_chip_back', exist_ok = True)
os.makedirs('train/cc_chip_front', exist_ok = True)
os.makedirs('train/ar', exist_ok = True)

os.makedirs('val/cm_front', exist_ok = True)
os.makedirs('val/cc_2_front', exist_ok = True)
os.makedirs('val/cm_back', exist_ok = True)
os.makedirs('val/cc_back', exist_ok = True)
os.makedirs('val/dl_front', exist_ok = True)
os.makedirs('val/cc_chip_back', exist_ok = True)
os.makedirs('val/cc_chip_front', exist_ok = True)
os.makedirs('val/ar', exist_ok = True)

os.makedirs('test/cm_front', exist_ok = True)
os.makedirs('test/cc_2_front', exist_ok = True)
os.makedirs('test/cm_back', exist_ok = True)
os.makedirs('test/cc_back', exist_ok = True)
os.makedirs('test/dl_front', exist_ok = True)
os.makedirs('test/cc_chip_back', exist_ok = True)
os.makedirs('test/cc_chip_front', exist_ok = True)
os.makedirs('test/ar', exist_ok = True)

count_json = 0
count_img = 0

for js in os.listdir(folder_path_train_image):
    if "cc_2_front" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cc_2_front')
    if "cc_back" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cc_back')
    if "cm_front" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cm_front')
    if "cm_back" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cm_back')
    if "cc_chip_back" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cc_chip_back')
    if "cc_chip_front" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/cc_chip_front')
    if "dl_front" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/dl_front')
    if "ar" in js:
        shutil.move(os.path.join(folder_path_train_image, js), './train/ar')

for js in os.listdir(folder_path_val_image):
    if "cc_2_front" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cc_2_front')
    if "cc_back" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cc_back')
    if "cm_front" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cm_front')
    if "cm_back" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cm_back')
    if "cc_chip_back" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cc_chip_back')
    if "cc_chip_front" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/cc_chip_front')
    if "dl_front" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/dl_front')
    if "ar" in js:
        shutil.move(os.path.join(folder_path_val_image, js), './val/ar')

for js in os.listdir(folder_path_test_image):
    if "cc_2_front" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cc_2_front')
    if "cc_back" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cc_back')
    if "cm_front" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cm_front')
    if "cm_back" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cm_back')
    if "cc_chip_back" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cc_chip_back')
    if "cc_chip_front" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/cc_chip_front')
    if "dl_front" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/dl_front')
    if "ar" in js:
        shutil.move(os.path.join(folder_path_test_image, js), './test/ar')