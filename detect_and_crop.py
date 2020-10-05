import cv2
import numpy as np 
import random
from random import randint
import os
import csv
import tensorflow as tf
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint

H = 144
W = 256
train_H = 72
train_W = 128
batch_size = 16
train_count = 100
validation_count = 100
dataset_size = 20

workdir = "/home/tharindu/Desktop/black/codes/Black/loclization/ckpt"
data_folder = "/home/1TB/retina_labeled/train"

def createIrisModel():
	model = models.Sequential()
	model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.MaxPooling2D((2, 2)))
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(64, activation='relu'))
	model.add(layers.Dense(2))

	ckpt = workdir + "/crop_eye.h5"
	if(os.path.isfile(ckpt)):
		tf.print("Loading existing wewights found inside: ", ckpt)
		model.load_weights(ckpt)
	else:
		tf.print("No wights found.")

	return model

def generateCroppedImages():
	count = 0
	iris_model = createIrisModel()
	while(count<20000000):
		count += 1
		image_count = 1
		break_flag = False
		batch = []
		labels = []
		while(True):
			iris_list = os.listdir(real_train_folder)
			random.shuffle(iris_list)
			for iris in iris_list:
				image_path = iris_folder + '/' + iris
				img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
				crop_copy = cv2.resize(img, (train_W,train_H), interpolation = cv2.INTER_AREA)
				iris_cordinates = iris_model.predict(tf.reshape(crop_copy, [1,train_W,train_H,1]))

				### edit from here
				batch.append(train_image)
				labels.append([iris_cordinates[0]/H/2,iris_cordinates[1]/W/2])
				image_count += 1
				if(image_count > batch_size):
					break_flag = True
					break;
			if(break_flag):
				break;
		yield (tf.reshape(batch, [batch_size,train_W,train_H,1]),tf.reshape(labels, [batch_size,2]))
		batch = []
		labels = []