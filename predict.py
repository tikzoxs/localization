import tensorflow as tf
import absl.logging as logging
import keras
from tensorflow.keras import datasets, layers, models
from keras.callbacks import ModelCheckpoint
import generate_dataset
import os
import cv2

H = 120
W = 160
train_H = 60
train_W = 80
batch_size = 16
TEST_ITERATIONS = 50

box_half = 48


workdir = "/home/tharindu/Desktop/black/codes/Black/loclization/ckpt"
tensorboard = "/home/tharindu/Desktop/black/codes/Black/loclization/tensorboard"

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(train_W, train_H, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(2))

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

#defining optimizers and loss funtction
# model.compile(optimizer=keras.optimizers.Adadelta(), 
              # loss=tf.keras.losses.MeanSquaredError(),
              # metrics=['mean_squared_error'])

#load weights if available
ckpt = workdir + "/fine_tune.h5"
if(os.path.isfile(ckpt)):
	tf.print("Loading existing wewights found inside: ", ckpt)
	model.load_weights(ckpt)
else:
	tf.print("No wights found.")


#print model
model.summary()

for i in range(TEST_ITERATIONS):
	# img, cordinates = generate_dataset.testingImageGenerator()
	img, cordinates = generate_dataset.readImagesFromTestFolder()
	img = cv2.resize(img, (W,H), interpolation = cv2.INTER_AREA)
	print(img.shape)
	image = cv2.resize(img, (train_W,train_H), interpolation = cv2.INTER_AREA)
	center = model.predict(tf.reshape(image, [1,train_W,train_H,1]))
	x = int(center[0][0]*H)
	y = int(center[0][1]*W)
	print([center[0][0]*H, center[0][1]*W], [cordinates[0]*H, cordinates[1]*W], "---------" , ((center[0][0] - cordinates[0])**2 + (center[0][1] - cordinates[1])**2)/2)
	display_image = cv2.circle(img, (int(center[0][1]*train_W),int(center[0][0]*train_H)), 32, (255,0,0), 2)
	cv2.imshow('prediction',display_image/255)
	x1 = max(0, x-box_half)
	x2 = min(120, x+box_half)
	y1 = max(0, y-box_half)
	y2 = min(160, y+box_half)
	print(x1,x2,y1,y2)
	cv2.imshow('crop',display_image[x1:x2, y1:y2])
	cv2.waitKey(0)



