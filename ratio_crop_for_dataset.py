import cv2 
import os
import random

break_flag = False

source_folder = '/home/1TB/EyeKnowYouSSLData/p'
target_folder = '/home/1TB/new_iris_dataset/train' 
# file_list = os.listdir(source_folder)
labeled_list_size = len(os.listdir(target_folder))
count = labeled_list_size
print("starting count : ", count)

crop_w = 640
crop_h = 480

def nothing(event, x, y, flags, param):
	print("inside nothing")

def crop_eye(event, x, y, flags, param): 
	global break_flag
	global u_image
	global count
	global target_folder
	global h,w,c
	global i

	if event == cv2.EVENT_MOUSEMOVE: 
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		if(h == 720):
			img = cv2.resize(img,(w*crop_h//h,crop_h))
		cv2.rectangle(img, (x - crop_w//2, y - crop_h//2), (x + crop_w//2, y + crop_h//2), (0, 255, 0), 1)
		# cv2.circle(img, (x, y), 25, (0, 255, 0), 1) 
		cv2.imshow("Select a good crop", img)
		# cv2.waitKey(10)

	if event == cv2.EVENT_LBUTTONDOWN:
		print("Press y key")
		response = None
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
		if(h == 720):
			img = cv2.resize(img,(w*crop_h//h,crop_h))
		cv2.rectangle(img, (x - crop_w//2, y - crop_h//2), (x + crop_w//2, y + crop_h//2), (255, 255, 0), 3)
		# cv2.circle(img, (x, y), 10, (255, 255, 0), 2)
		cv2.imshow("Select a good crop", img)
		while(response == None):
			response = cv2.waitKey()
		if response == ord('y'): 
			print("crop registered")
			count += 1
			img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
			if(h == 720):
				img = cv2.resize(img,(w*crop_h//h,crop_h))
			# if(x>63 and y>63 and x<w-64 and y<h-64):
			# 	print("if true")
			cropped_image = img[0 : crop_h, x - (crop_w//2) : x + (crop_w//2)]
			cv2.imwrite(target_folder + '/' + str(count) + ".jpg",cropped_image)
			i += 1
			print("Press y key again")
			cv2.setMouseCallback("Select a good crop", nothing)
			break_flag = True
		else:
			print("Click again")
			break_flag = False


  
cv2.namedWindow(winname = "Select a good crop") 
 
i = 0
while(i<200):
# for i in range(10):
	user = random.randint(1, 14)
	frame = random.randint(1, 150000)
	filename = source_folder + str(user) + '/' + str(frame) + '.jpg'
	if(os.path.isfile(filename)):
		u_image = filename
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		h,w = (img.shape)
		if(h == 720):
			print("Move the mouse over the image")
			cv2.setMouseCallback("Select a good crop", crop_eye)
			while True: 
				if break_flag:
					break
				if cv2.waitKey(2) & 0xFF == 27: 
					break
		else:
			print("correct sized image. saved at once!")
			cv2.imwrite(target_folder + '/' + str(count) + ".jpg",img)
			i += 1
		break_flag = False
		
	# else:
	# 	i -= 1
	print("****************************")
		
cv2.destroyAllWindows()