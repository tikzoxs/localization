import cv2
import os

folder = "/home/1TB/new_iris_dataset/normalized"

frame__list = os.listdir(folder)
for frame in frame__list:
	image_number = int(frame.split('.')[0].split('_')[0])
	if(image_number == 1240):
		image_path = folder + '/' + frame
		img = cv2.imread(image_path)
		center = [int(frame.split('.')[0].split('_')[2]), int(frame.split('.')[0].split('_')[1])]
		print(center)
		display_image = cv2.circle(img, (int(center[1]),int(center[0])), 32, (255,0,0), 2)
		cv2.imshow('visualize',display_image/255)
		cv2.waitKey()