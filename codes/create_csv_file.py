# version
# Tensorflow 1.14.0 + Keras 2.2.4 + Python 3.6

import os
import shutil
import random
import glob
import pandas as pd


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


# Step1. 데이터 전처리(train, test, val)
if (0):
    mkdir("모듈8데이터(SSD_앵무새)/train/")
    mkdir("모듈8데이터(SSD_앵무새)/test/")
    mkdir("모듈8데이터(SSD_앵무새)/val/")
    train_csv, test_csv, val_csv = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for filename in glob.iglob('모듈8데이터(SSD_앵무새)/**/*.csv', recursive=True):
        csv = pd.read_csv(filename)
        if 'train' in filename:
            train_csv = train_csv.append(csv)
            for img in glob.iglob('모듈8데이터(SSD_앵무새)/**/*.jpg', recursive=True):
                if img.split("\\")[-1] in csv['frame'].values:
                    shutil.copy(img, "모듈8데이터(SSD_앵무새)/train/")
        if 'test' in filename:
            test_csv = test_csv.append(csv)
            for img in glob.iglob('모듈8데이터(SSD_앵무새)/**/*.jpg', recursive=True):
                if img.split("\\")[-1] in csv['frame'].values:
                    shutil.copy(img, "모듈8데이터(SSD_앵무새)/test/")
        if 'val' in filename:
            val_csv = val_csv.append(csv)
            for img in glob.iglob('모듈8데이터(SSD_앵무새)/**/*.jpg', recursive=True):
                if img.split("\\")[-1] in csv['frame'].values:
                    shutil.copy(img, "모듈8데이터(SSD_앵무새)/val/")

    train_csv.to_csv('모듈8데이터(SSD_앵무새)/train/train.csv')
    test_csv.to_csv('모듈8데이터(SSD_앵무새)/test/test.csv')
    val_csv.to_csv('모듈8데이터(SSD_앵무새)/val/val.csv')


