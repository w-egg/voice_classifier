import argparse
import cv2
import os
import shutil
import sys

SAVE_PATH = './face/'

FALG = None

CASCADES = ["default", "alt"]
global face_detect_count
face_detect_count = 0


def save_image(img, file_name, cascade):
    global face_detect_count
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 50))
    if len(face) > 0:
        for rect in face:
            save_dir = SAVE_PATH + dir_name
            if not os.path.isdir(save_dir):
                os.mkdir(SAVE_PATH + dir_name)
            save_file = save_dir + '/' + \
                file_name.split('.')[0] + '_' + \
                str(face_detect_count) + '.jpg'
            cv2.imwrite(save_file,
                        img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]])
            face_detect_count += 1
            print(save_file)


def face_cut(dir_name, cascade):
    files = os.listdir(dir_name)
    for file_name in files:
        if os.path.isdir(dir_name + '/' + file_name):
            continue
        img = cv2.imread(dir_name + '/' + file_name)
        if img is None:
            continue
        save_image(img, file_name, cascade)


if __name__ == '__main__':
    print("フォルダ名を指定してください")
    dir_name = input()
    if not os.path.exists(dir_name):
        sys.exit()
    for cascade in CASCADES:
        cascade_path = f'./cascade/haarcascade_frontalface_{cascade}.xml'
        if not os.path.exists(cascade_path):
            continue
        face_cascade = cv2.CascadeClassifier(cascade_path)
        face_cut(dir_name, face_cascade)
