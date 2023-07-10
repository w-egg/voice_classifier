import os


def add_directory(dir: str):
    dir_arr = []
    for dir_name in os.listdir(dir):
        dir_arr.append(dir_name)
        print(dir_name)
    return dir_arr
