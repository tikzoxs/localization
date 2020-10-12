import cv2 
import os
import random
import numpy as np

W = 160
H = 120
dimensions = [H,W]
eye_radius = 30

def mirrorImage(image):
	return cv2.flip(image, 1)

# def zoomCrop(image):


def randomOffset(image, iris_cordinates):
	axis = random.randint(0, 1)
	direction = random.randint(0, 1)
	# print(dimensions[axis] - iris_cordinates[axis] - eye_radius, max(iris_cordinates[axis] - eye_radius, 0))
	max_offset = min(dimensions[axis] - iris_cordinates[axis] - eye_radius, max(iris_cordinates[axis] - eye_radius, 0))
	if(max_offset <= 0):
		offset_amount = 0
	else:
		offset_amount = random.randint(1, max_offset)
	# print(axis,direction,dimensions[axis],iris_cordinates,max_offset,offset_amount)
	dummy = np.ones(image.shape)*150 # (120,160)
	if(axis == 0):
		np.copyto(dummy[offset_amount*direction: H-offset_amount*(1-direction), :], image[offset_amount*(1-direction): H-offset_amount*direction, :])
	if(axis == 1):
		np.copyto(dummy[:, offset_amount*direction: W-offset_amount*(1-direction)], image[:, offset_amount*(1-direction): W-offset_amount*direction])
	return dummy, offset_amount, axis, direction


def augmentFolder(source_folder, target_folder, mode=cv2.IMREAD_GRAYSCALE):
	image_list = os.listdir(source_folder)
	count = 0
	for k in range(3):
		for frame in image_list:
			count += 1
			image_path = source_folder + '/' + frame
			image = cv2.imread(image_path, mode)
			iris_cordinates = [int(frame.split('.')[0].split('_')[2]), int(frame.split('.')[0].split('_')[1])]
			image_number = frame.split('.')[0].split('_')[0]
			offset_image, offset_amount, axis, direction = randomOffset(image, iris_cordinates)
			iris_cordinates[axis] = iris_cordinates[axis] + (-1 + 2*direction)*offset_amount
			# mirror = mirrorImage(image)
			cv2.imwrite(target_folder + '/' + image_number + '_' +  str(iris_cordinates[1]) + '_' + str(iris_cordinates[0]) + '.jpg', offset_image)
			# exit()
			# if(count == 50):
			# 	exit()

augmentFolder("/home/1TB/new_iris_dataset/normalized", "/home/1TB/new_iris_dataset/offset")