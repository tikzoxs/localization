import cv2
import os

folder = "/home/1TB/new_iris_dataset/normalized"
data_folder = "/home/1TB/EyeKnowYouSSLData"

################## To check iris annotations ##################
# frame__list = os.listdir(folder)
# for frame in frame__list:
# 	image_number = int(frame.split('.')[0].split('_')[0])
# 	if(image_number == 1240):
# 		image_path = folder + '/' + frame
# 		img = cv2.imread(image_path)
# 		center = [int(frame.split('.')[0].split('_')[2]), int(frame.split('.')[0].split('_')[1])]
# 		print(center)
# 		display_image = cv2.circle(img, (int(center[1]),int(center[0])), 32, (255,0,0), 2)
# 		cv2.imshow('visualize',display_image/255)
# 		cv2.waitKey()
###############################################################

############# To check image shapes #####################
user_list = os.listdir(data_folder)
for user in user_list:
	folder_path = data_folder + '/' + user
	image_list = os.listdir(folder_path)
	image_path = folder_path + '/' + image_list[50]
	image = cv2.imread(image_path,0)
	s = image.shape
	# print(s)
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
	else:
		continue
	print(user, image.shape)
	cv2.imwrite(target_folder + '/' + str(count) + "_" + str(x) + "_" + str(y) + ".jpg",img)
	cv2.imshow('image',image)
	cv2.waitKey(0)