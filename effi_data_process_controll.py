import os
import shutil

import tqdm
from PIL import Image
import json
import random
from PIL import ImageDraw, ImageFont
from matplotlib.font_manager import json_load
import argparse

folder_path_train_image = 'train/image'
folder_path_val_image = 'val/image'
folder_path_test_image = 'test/image'

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

count_train_1 = 0
count_train_2 = 0
count_train_3 = 0
count_train_4 = 0
count_train_5 = 0
count_train_6 = 0
count_train_7 = 0

count_val_1 = 0
count_val_2 = 0
count_val_3 = 0
count_val_4 = 0
count_val_5 = 0
count_val_6 = 0
count_val_7 = 0

count_test_1 = 0
count_test_2 = 0
count_test_3 = 0
count_test_4 = 0
count_test_5 = 0
count_test_6 = 0
count_test_7 = 0

number_of_train = 1000
number_of_val = 120
number_of_test = 120

parser = argparse.ArgumentParser()
parser.add_argument('--number_of_train', required=True, type=int)
parser.add_argument('--number_of_val', required=True, type=int)
parser.add_argument('--number_of_test', required=True, type=int)
number_of_train = vars(parser.parse_args())['number_of_train']
number_of_val = vars(parser.parse_args())['number_of_val']
number_of_test = vars(parser.parse_args())['number_of_test']

for js in os.listdir(folder_path_train_image):
    if "cc_2_front" in js:
        if count_train_1 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cc_2_front')
            count_train_1 +=1
    if "cc_back" in js:
        if count_train_2 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cc_back')
            count_train_2 +=1
    if "cm_front" in js:
        if count_train_3 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cm_front')
            count_train_3 +=1
    if "cm_back" in js:
        if count_train_4 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cm_back')
            count_train_4 +=1
    if "cc_chip_back" in js:
        if count_train_5 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cc_chip_back')
            count_train_5 +=1
    if "cc_chip_front" in js:
        if count_train_6 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/cc_chip_front')
            count_train_6 +=1
    if "dl_front" in js:
        if count_train_7 <= number_of_train:
            shutil.move(os.path.join(folder_path_train_image, js), './train/dl_front')
            count_train_7 +=1

for js in os.listdir(folder_path_val_image):
    if "cc_2_front" in js:
        if count_val_1 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cc_2_front')
            count_val_1 +=1
    if "cc_back" in js:
        if count_val_2 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cc_back')
            count_val_2 +=1
    if "cm_front" in js:
        if count_val_3 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cm_front')
            count_val_3 +=1
    if "cm_back" in js:
        if count_val_4 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cm_back')
            count_val_4 +=1
    if "cc_chip_back" in js:
        if count_val_5 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cc_chip_back')
            count_val_5 +=1
    if "cc_chip_front" in js:
        if count_val_6 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/cc_chip_front')
            count_val_6 +=1
    if "dl_front" in js:
        if count_val_7 <=number_of_val:
            shutil.move(os.path.join(folder_path_val_image, js), './val/dl_front')
            count_val_7 +=1

for js in os.listdir(folder_path_test_image):
    if "cc_2_front" in js:
        if count_test_1 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cc_2_front')
            count_test_1 +=1
    if "cc_back" in js:
        if count_test_2 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cc_back')
            count_test_2 +=1
    if "cm_front" in js:
        if count_test_3 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cm_front')
            count_test_3 +=1
    if "cm_back" in js:
        if count_test_4 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cm_back')
            count_test_4 +=1
    if "cc_chip_back" in js:
        if count_test_5 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cc_chip_back')
            count_test_5 +=1
    if "cc_chip_front" in js:
        if count_test_6 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/cc_chip_front')
            count_test_6 +=1
    if "dl_front" in js:
        if count_test_7 <=number_of_test:
            shutil.move(os.path.join(folder_path_test_image, js), './test/dl_front')
            count_test_7 +=1
