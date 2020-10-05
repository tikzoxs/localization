import cv2 
import os
import random

break_flag = False
source_folder = '/home/1TB/new_iris_dataset/train'
target_folder = '/home/1TB/new_iris_dataset/annotated'  
# file_list = os.listdir(source_folder)
labeled_list_size = len(os.listdir(target_folder))
count = labeled_list_size
read_count = count

def nothing(event, x, y, flags, param):
	print("inside nothing")

def crop_eye(event, x, y, flags, param): 
	global break_flag
	global u_image
	global count
	global target_folder
	global h,w,c

	if event == cv2.EVENT_MOUSEMOVE: 
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		img = cv2.resize(img,(160,120))
		cv2.circle(img, (x, y), 25, (255, 255, 0), 1)
		# cv2.circle(img, (x, y), 25, (0, 255, 0), 1) 
		cv2.imshow("Click the center of the Eye", img)
		# cv2.waitKey(10)

	if event == cv2.EVENT_LBUTTONDOWN:
		print("Press y key")
		response = None
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
		img = cv2.resize(img,(160,120))
		cv2.circle(img, (x, y), 25, (255, 255, 0), 2)
		# cv2.circle(img, (x, y), 10, (255, 255, 0), 2)
		cv2.imshow("Click the center of the Eye", img)
		while(response == None):
			response = cv2.waitKey()
		if response == ord('y'): 
			print("crop registered")
			count += 1
			img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE)
			img = cv2.resize(img,(160,120))
			print(x,y)
			cv2.imwrite(target_folder + '/' + str(count) + "_" + str(x) + "_" + str(y) + ".jpg",img)
			print("Press y key again")
			cv2.setMouseCallback("Click the center of the Eye", nothing)
			break_flag = True
		else:
			print("Click again")
			break_flag = False

  
cv2.namedWindow(winname = "Click the center of the Eye") 
 
for f in range(read_count, len(os.listdir(source_folder))):
	filename = source_folder + '/' + str(f) + '.jpg' 
	if(os.path.isfile(filename)):
		u_image = filename
		img = cv2.imread(u_image, cv2.IMREAD_GRAYSCALE) 
		print("Move the mouse over the image")
		cv2.setMouseCallback("Click the center of the Eye", crop_eye)
		while True: 
			if break_flag:
				break
			if cv2.waitKey(2) & 0xFF == 27: 
				break
		break_flag = False
		
cv2.destroyAllWindows()