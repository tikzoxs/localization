import tensorflow as tf
import absl.logging as logging
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint
import generate_dataset
import os
import cv2
import random

H = 480
W = 640
train_H = 60
train_W = 80
batch_size = 16
TEST_ITERATIONS = 3

box_half = 96


workdir = "/home/tharindu/Desktop/black/codes/Black/loclization/ckpt"
data_folder = "/home/1TB/EyeKnowYouSSLData"

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


def crop(image):
	display_image = cv2.resize(image, (W,H), interpolation = cv2.INTER_AREA)
	pred_input = cv2.resize(display_image, (train_W,train_H), interpolation = cv2.INTER_AREA)
	center = model.predict(tf.reshape(pred_input, [1,train_W,train_H,1]))
	x = int(center[0][0]*H)
	y = int(center[0][1]*W)
	x1 = max(0, x-box_half)
	x2 = min(H, x+box_half)
	y1 = max(0, y-box_half)
	y2 = min(W, y+box_half)
	return display_image[x1:x2, y1:y2]

count = 0
user_list = os.listdir(data_folder)
random.shuffle(user_list)
for  user in user_list:
	folder_path = data_folder + '/' + user
	image_list = os.listdir(folder_path)
	random.shuffle(image_list)
	for image in image_list:
		image_path = folder_path + '/' + image
		image = cv2.imread(image_path,0)
		cropped = crop(image)
		cv2.imshow('original',image)
		cv2.imshow('crop',cropped)
		cv2.waitKey(0)
		count += 1
		if(count == TEST_ITERATIONS):
			break



