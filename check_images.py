import cv2
import os

folder = "/home/1TB/EyeKnowYouSSLData"

folder_list = os.listdir(folder)
for user in folder_list:
	images = os.listdir(folder + '/' + user )
	image_path = folder + '/' + user + '/' + images[50000]
	img = cv2.imread(image_path)
	print(user,img.shape)