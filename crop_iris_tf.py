import tensorflow as tf
import absl.logging as logging
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint
import generate_dataset
import os
import cv2
import random
import numpy as np

H = 480
W = 640
train_H = 60
train_W = 80
batch_size = 128
TEST_ITERATIONS = 100

box_half = 96


workdir = "./ckpt"
data_folder = "/home/1TB/new_users_by_number/"
target_folder = "/home/1TB/Cropped_new_users"

#2 layer model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(2))


#load weights if available
ckpt = workdir + "/fine_tune.h5"
if(os.path.isfile(ckpt)):
	tf.print("Loading existing wewights found inside: ", ckpt)
	model.load_weights(ckpt)
else:
	tf.print("No wights found.")
	exit()


def crop(gray_image_list):
	centers = model.predict(tf.reshape(gray_image_list, [batch_size,train_W,train_H,1]))
	cordinates = []
	for center in centers:
		x = int(center[0]*H)
		y = int(center[1]*W)
		x1 = max(0, x-box_half)
		x2 = min(H, x+box_half)
		y1 = max(0, y-box_half)
		y2 = min(W, y+box_half)
		cordinates.append([x1,x2,y1,y2])
	return cordinates

def userBasedCrop(image, user):
	s = image.shape
	image = cv2.resize(image, (s[1]*480//s[0],480), interpolation = cv2.INTER_AREA)
	if(user == 'p2'):
		image = image[:, 120: 120+640]
	elif(user == 'p5'):
		image = image[:, 0: 0+640]
	elif(user == 'p14'):
		image = image[:, 105: 105+640]
	elif(user == 'p9'):
		image = image[:, 105: 105+640]
	elif(user == 'p8'):
		image = image[:, 60: 60+640]
	elif(user == 'p13'):
		image = image[:, 71: 71+640]
	return image

global_count = 0
user_list = os.listdir(data_folder)
for user in user_list:
	count = 0
	folder_path = data_folder + '/' + user
	image_list = os.listdir(folder_path)
	os.mkdir(target_folder + '/' + user)
	colour_list = []
	gray_list = []
	name_list = []
	for image in image_list:
		count += 1
		global_count += 1
		image_path = folder_path + '/' + image
		image_colour = cv2.imread(image_path,1)
		sized_colour_image = userBasedCrop(image_colour, user)
		grayscale = cv2.cvtColor(sized_colour_image, cv2.COLOR_BGR2GRAY)
		gray_list.append(cv2.resize(grayscale, (train_W,train_H), interpolation = cv2.INTER_AREA))
		colour_list.append(sized_colour_image)
		name_list.append(image)
		if(count % batch_size == 0):
			cordinates = crop(gray_list)
			for cordinate,sci,name in zip(cordinates,colour_list,name_list):
				x1,x2,y1,y2 = cordinate
				cropped = cv2.resize(sci[x1:x2, y1:y2], (96,96), interpolation = cv2.INTER_AREA)
				cv2.imwrite(target_folder + '/' + user + '/' + name, cropped)
			colour_list = []
			gray_list = []
			name_list = []
			print(global_count)
		# cv2.imshow('original',sized_colour_image)
		# cv2.imshow('crop',cropped)
		# cv2.waitKey(0)
		# count += 1
		# if(count > TEST_ITERATIONS):
		# 	exit()



