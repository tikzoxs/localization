import cv2
import numpy as np 
import random
from random import randint
import os
import csv
import tensorflow as tf

BASE_BRIGTNESS = 120
FILL_BASE_COLOR = 150
LINE_BASE_COLOR = 50
EYE_RADIUS = 32
REQUIRED_COUNT = 5000
GAUSSIAN_BLUR_RADIUS = [0,3,5,7,9]
H = 144
W = 256
train_H = 72
train_W = 128
batch_size = 16
train_count = 100
validation_count = 100
dataset_size = 20


iris_folder = "/home/1TB/retina_labeled/train"
real_train_folder = "/home/1TB/retina_labeled/train"
real_validation_folder = "/home/1TB/retina_labeled/validation"
real_test_folder = "/home/1TB/retina_labeled/test"
train_output_folder = "/home/1TB/retina_labeled/generated"
validation_output_folder = "/home/1TB/retina_labeled/generated"

def randomBackground(h,w):
	return np.random.randint(256, size=(h, w))

def plannedBackground(h,w,brightness):
	image = np.asarray([[float(brightness)]*w]*h)
	for c in range(randint(1,10)):
		image = cv2.circle(image, (randint(0,W),randint(0,H)), randint(40,120), randint(FILL_BASE_COLOR-20,FILL_BASE_COLOR+20), -1)
	image = cv2.GaussianBlur(image,(25,25),cv2.BORDER_DEFAULT)
	for c in range(randint(1,10)):
		image = cv2.circle(image, (randint(0,W),randint(0,H)), randint(50,500), randint(LINE_BASE_COLOR-20,LINE_BASE_COLOR+20), randint(1,4))
	return image
	


def cropAndInsertIris(image,back,iris_cordinates,new_iris_cordinates):
	# print(iris_cordinates,new_iris_cordinates)
	for x in range(-1*EYE_RADIUS + 1, EYE_RADIUS - 1):
		for y in range(-1*EYE_RADIUS + 1, EYE_RADIUS - 1):
			if(x**2 + y**2 < EYE_RADIUS**2):
				ic1 = min(iris_cordinates[0]+x,143)
				ic2 = min(iris_cordinates[1]+y,255)
				nic1 = min(new_iris_cordinates[0]+x,143)
				nic2 = min(new_iris_cordinates[1]+y,255)
				back[nic1, nic2] = image[ic1, ic2]
	blur_kernel = (GAUSSIAN_BLUR_RADIUS[randint(0,2)],GAUSSIAN_BLUR_RADIUS[randint(0,2)])	
	return cv2.GaussianBlur(back,blur_kernel,cv2.BORDER_DEFAULT)


def generateImages():
	# with open('/home/1TB/retina_labeled/generated/validation.csv', 'w', newline='') as file:
	# 	fieldnames = ['image_name']
	# 	writer = csv.DictWriter(file, fieldnames=fieldnames)
	count = 0
	while(count<20000000):
		count += 1
		image_count = 1
		break_flag = False
		batch = []
		labels = []
		while(True):
			iris_list = os.listdir(iris_folder)
			random.shuffle(iris_list)
			for iris in iris_list:
				# print(image_count)
				image_path = iris_folder + '/' + iris
				img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
				iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
				new_iris_cordinates = [randint(1.5*EYE_RADIUS, H - 1.5*EYE_RADIUS), randint(1.5*EYE_RADIUS, W - 1.5*EYE_RADIUS)]
				generated_image = cropAndInsertIris(img,plannedBackground(H,W,randint(BASE_BRIGTNESS-50,BASE_BRIGTNESS+50)),iris_cordinates,new_iris_cordinates)
				generated_image = cv2.resize(generated_image, (train_W,train_H), interpolation = cv2.INTER_AREA)
				# cv2.imwrite(validation_output_folder + '/' + str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg', generated_image)
				# writer.writerow({'image_name': str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg'})
				batch.append(generated_image)
				labels.append([new_iris_cordinates[0]/H/2,new_iris_cordinates[1]/W/2])
				image_count += 1
				if(image_count > batch_size):
					break_flag = True
					break;
			if(break_flag):
				break;
		yield (tf.reshape(batch, [batch_size,train_W,train_H,1]),tf.reshape(labels, [batch_size,2]))
		batch = []
		labels = []

def generateImagesValidation():
	# with open('/home/1TB/retina_labeled/generated/validation.csv', 'w', newline='') as file:
	# 	fieldnames = ['image_name']
	# 	writer = csv.DictWriter(file, fieldnames=fieldnames)
	count = 0
	while(count<1000000):
		count += 1
		image_count = 1
		break_flag = False
		batch = []
		labels = []
		while(True):
			iris_list = os.listdir(iris_folder)
			random.shuffle(iris_list)
			for iris in iris_list:
				# print(image_count)
				image_path = iris_folder + '/' + iris
				img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
				iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
				new_iris_cordinates = [randint(1.5*EYE_RADIUS, H - 1.5*EYE_RADIUS), randint(1.5*EYE_RADIUS, W - 1.5*EYE_RADIUS)]
				generated_image = cropAndInsertIris(img,plannedBackground(H,W,randint(BASE_BRIGTNESS-50,BASE_BRIGTNESS+50)),iris_cordinates,new_iris_cordinates)
				generated_image = cv2.resize(generated_image, (train_W,train_H), interpolation = cv2.INTER_AREA)
				# cv2.imwrite(validation_output_folder + '/' + str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg', generated_image)
				# writer.writerow({'image_name': str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg'})
				batch.append(generated_image)
				labels.append([new_iris_cordinates[0]/H/2,new_iris_cordinates[1]/W/2])
				image_count += 1
				if(image_count > batch_size):
					break_flag = True
					break;
			if(break_flag):
				break;
		yield (tf.reshape(batch, [batch_size,train_W,train_H,1]),tf.reshape(labels, [batch_size,2]))
		batch = []
		labels = []
	

def testingImageGenerator():
	iris_list = os.listdir(iris_folder)
	random.shuffle(iris_list)
	iris = iris_list[0]
	image_path = iris_folder + '/' + iris
	img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
	iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
	new_iris_cordinates = [randint(1.5*EYE_RADIUS, H - 1.5*EYE_RADIUS), randint(1.5*EYE_RADIUS, W- 1.5*EYE_RADIUS)]
	generated_image = cropAndInsertIris(img,plannedBackground(H,W,randint(BASE_BRIGTNESS-50,BASE_BRIGTNESS+50)),iris_cordinates,new_iris_cordinates)
	# cv2.imwrite(validation_output_folder + '/' + str(randint(1,10)) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg', generated_image)
	# generated_image = cv2.resize(generated_image, (train_W,train_H), interpolation = cv2.INTER_AREA)
		# cv2.imwrite(validation_output_folder + '/' + str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg', generated_image)
		# writer.writerow({'image_name': str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg'})
	return generated_image, [new_iris_cordinates[0]/H/2,new_iris_cordinates[1]/W/2]

def generateImagesAndSave():
	image_count = 1
	break_flag = False

	while(True):
		iris_list = os.listdir(iris_folder)
		random.shuffle(iris_list)
		for iris in iris_list:
			image_path = iris_folder + '/' + iris
			img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
			iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
			new_iris_cordinates = [randint(1.5*EYE_RADIUS, H - 1.5*EYE_RADIUS), randint(1.5*EYE_RADIUS, W - 1.5*EYE_RADIUS)]
			generated_image = cropAndInsertIris(img,plannedBackground(H,W,randint(BASE_BRIGTNESS-50,BASE_BRIGTNESS+50)),iris_cordinates,new_iris_cordinates)
			generated_image = cv2.resize(generated_image, (train_W,train_H), interpolation = cv2.INTER_AREA)
			cv2.imwrite(validation_output_folder + '/' + str(image_count) + '_' + str(new_iris_cordinates[0]) + '_' + str(new_iris_cordinates[1]) + '.jpg', generated_image)
			# label = [new_iris_cordinates[0]/H/2,new_iris_cordinates[1]/W/2]
			# print(label)
			# print(tf.reshape(label+label+label, [3,2]))
			image_count += 1
			if(image_count > dataset_size):
				break_flag = True
				break;
		if(break_flag):
			break;

def readImagesFromTrainFolder():
	# with open('/home/1TB/retina_labeled/generated/validation.csv', 'w', newline='') as file:
	# 	fieldnames = ['image_name']
	# 	writer = csv.DictWriter(file, fieldnames=fieldnames)
	count = 0
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
				# print(image_count)
				image_path = iris_folder + '/' + iris
				img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
				iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
				train_image = cv2.resize(img, (train_W,train_H), interpolation = cv2.INTER_AREA)
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

def readImagesFromValidationFolder():
	# with open('/home/1TB/retina_labeled/generated/validation.csv', 'w', newline='') as file:
	# 	fieldnames = ['image_name']
	# 	writer = csv.DictWriter(file, fieldnames=fieldnames)
	count = 0
	while(count<20000000):
		count += 1
		image_count = 1
		break_flag = False
		batch = []
		labels = []
		while(True):
			iris_list = os.listdir(real_validation_folder)
			random.shuffle(iris_list)
			for iris in iris_list:
				# print(image_count)
				image_path = real_validation_folder + '/' + iris
				img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
				iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
				train_image = cv2.resize(img, (train_W,train_H), interpolation = cv2.INTER_AREA)
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

def readImagesFromTestFolder():
	iris_list = os.listdir(real_test_folder)
	random.shuffle(iris_list)
	for iris in iris_list:
		image_path = real_test_folder + '/' + iris
		img = cv2.imread(image_path,cv2.COLOR_BGR2GRAY)
		iris_cordinates = [int(iris.split('.')[0].split('_')[2]), int(iris.split('.')[0].split('_')[1])]
		test_image = cv2.resize(img, (train_W,train_H), interpolation = cv2.INTER_AREA)

	return test_image, [iris_cordinates[0]/H/2,iris_cordinates[1]/W/2]






generateImagesAndSave()