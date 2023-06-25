import os
import time
import io
import hashlib
import signal
import requests
from PIL import Image
from selenium import webdriver
import pandas as pd

import json
import argparse
import numpy as np

import os
import cv2


def arreglarFolder():
    two_names = ['alfa', 'romeo', 'aston', 'martin', 'land', 'rover']
    # Path to the directory containing the image files
    path = "E:/TFM/Dataset1Car/images/train/"
    path2 = "E:/TFM/Dataset1Car/images/trainSmall/"

    # Get a list of all the image files in the directory
    folder_files = [f for f in os.listdir(path)]

    # Create a new directory to store the renamed images
    new_folder_path = path2
    os.makedirs(new_folder_path, exist_ok=True)

    make_set = set()
    b_model_file = ""
    # Loop through each image file and rename it
    for name_file in folder_files:
        name_file = name_file.lower()

        model = name_file.split(" ")[-1]
        make = name_file.replace(model, "")

        if make not in make_set:
            make_set.add(make)

            image_files = [f for f in os.listdir(path + name_file) if f.endswith(".jpg") or f.endswith(".png")]
            for image_file in image_files:
                # Load the image using cv2
                image_file = image_file.lower()
                image_path = os.path.join(path + name_file, image_file)
                image = cv2.imread(image_path)

                if name_file.lower().split()[0] in two_names and name_file.lower().split()[1] in two_names:
                    n = name_file.lower().split()
                    name_file_new = n[0] + " " + n[1] + " " + n[2]
                else:
                    n = name_file.lower().split()
                    name_file_new = n[0] + " " + n[1]

                os.makedirs(os.path.join(path2, name_file_new), exist_ok=True)
                # Rename the image file using a new naming convention
                new_image_file = name_file_new + "/" + image_file

                # Save the renamed image to the new directory
                new_image_path = os.path.join(new_folder_path, new_image_file)
                cv2.imwrite(new_image_path, image)

            if b_model_file != "":
                image_files = [f for f in os.listdir(path + b_model_file) if f.endswith(".jpg") or f.endswith(".png")]
                for image_file in image_files:
                    # Load the image using cv2
                    image_file = image_file.lower()
                    image_path = path + b_model_file + "/" + image_file
                    image = cv2.imread(image_path)

                    if name_file.lower().split()[0] in two_names and b_model_file.lower().split()[1] in two_names:
                        n = b_model_file.lower().split()
                        name_file_new = n[0] + " " + n[1] + " " + n[2]
                    else:
                        n = b_model_file.lower().split()
                        name_file_new = n[0] + " " + n[1]

                    os.makedirs(os.path.join(path2, name_file_new), exist_ok=True)
                    # Rename the image file using a new naming convention
                    new_image_file = name_file_new + "/" + image_file

                    # Save the renamed image to the new directory
                    new_image_path = os.path.join(new_folder_path, new_image_file)
                    cv2.imwrite(new_image_path, image)

        b_model_file = name_file
    print("All images have been renamed and saved to the new directory.")


if __name__ == '__main__':
    arreglarFolder()
