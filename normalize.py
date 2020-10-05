import cv2 
import os
import random
import numpy as np

def standardizeImage(image):
    # axis param denotes axes along which mean & std reductions are to be performed
    mn = np.mean(image)
    std = np.std(image)
    return (image - mn) / std * 255

def normalizeImage(image):
	array = np.zeros(image.shape)
	return cv2.normalize(image, array,  0, 180, cv2.NORM_MINMAX)

def normalizeFolder(source_folder, target_folder, mode=cv2.IMREAD_GRAYSCALE):
	image_list = os.listdir(source_folder)
	for frame in image_list:
		image_path = source_folder + '/' + frame
		image = cv2.imread(image_path, mode)
		normalized_image = normalizeImage(image)
		cv2.imwrite(target_folder + '/' + frame, normalized_image)

def standardizeFolder(source_folder, target_folder, mode=cv2.IMREAD_GRAYSCALE):
	image_list = os.listdir(source_folder)
	for frame in image_list:
		image_path = source_folder + '/' + frame
		image = cv2.imread(image_path, mode)
		normalized_image = standardizeImage(image)
		cv2.imwrite(target_folder + '/' + frame, normalized_image)

# normalizeFolder("/home/1TB/new_iris_dataset/annotated", "/home/1TB/new_iris_dataset/normalized_light")
standardizeFolder("/home/1TB/new_iris_dataset/annotated", "/home/1TB/new_iris_dataset/standardized")


