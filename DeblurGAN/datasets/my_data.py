import cv2
import os
import numpy as np

def motion_blur(image, kernel_size):
    # crate kernel
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel /= kernel_size

    # Apply blur
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

def blur_images_in_directory(input_dir, output_dir, kernel_size=15):
    # for every file in the dir
    for file_name in os.listdir(input_dir):
        print(file_name)
        # Read image
        image_path = os.path.join(input_dir, file_name)
        image = cv2.imread(image_path)

        # Apply blur
        blurred = motion_blur(image, kernel_size)

        # Save new image
        output_path = os.path.join(output_dir, file_name)
        cv2.imwrite(output_path, blurred)

# CHANGE THESE FOR WHERE DATASET IS ----------------------------
input_directory = "./my_data/"
output_directory = "./my_data_b/"
kernel_size = 25

blur_images_in_directory(input_directory, output_directory, kernel_size)
