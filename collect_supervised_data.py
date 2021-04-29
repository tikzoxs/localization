########### Label Mapping ###########
#       Monitor - Document = 1      #
#       Monotor - Media    = 2      #
#       Laptop  - Document = 3      #
#       Laptop  - Media    = 4      #
#       Phone   - Document = 5      #
#       Phone   - Media    = 6      #
#       Books              = 7      #
#       People             = 8      #
#       Blink              = 9      #
#       Other              = 0      #
#####################################



import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import h5py
import argparse
import datetime
import platform
import os
import numpy as np 
import cv2

#*********constants**********#
#original frame dimensions
ORIGINAL_HEIGHT = 480
ORIGINAL_WIDTH = 640
ORIGINAL_CHANNLES =  3
#cropped frame dimensions
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 90
#focusing camera
AUTO_FOCUS_1 = 'uvcdynctrl -d video'
AUTO_FOCUS_2 = ' --set=\'Focus, Auto\' '
SET_FOCUS_1 = 'uvcdynctrl --device=video'
SET_FOCUS_2 = ' --set=\'Focus (absolute)\' '
FOCUL_CONTINUITY = 5
FOCUS_ATTEMPTS = 5
BLUR_ERROR = 3
PRESET_OPTIMAL_FOCUS = 51
max_sharpness = 0
best_focus = PRESET_OPTIMAL_FOCUS
#other
DATAPATH = "/home/1TB/EyeKnowYou_SupervisedData"
#****************************#

#*********variables**********#
task_id = 0
#****************************#

#**********methods***********#
def checkDevice():
	device_name = platform.node()
	processor_architecture = platform.machine()
	print("\nEyeKnowYou data collection session initiated...")
	print("Checking System Iformation...")
	print("Computer Name:" + str(device_name))
	print("Instruction Architechture:" + str(processor_architecture))
	print("*******************************************")
	device_compatibility = True #change the logic later *****
	return device_compatibility

def identifyCorrectCamera():
	return 2

def disableDefaultAutofocus(camera):
	auto_focus = 0
	AUTO_FOCUS = AUTO_FOCUS_1 + str(camera) + AUTO_FOCUS_2 + str(auto_focus)
	response = os.popen(AUTO_FOCUS).read()
	print("------------------------------------------------------")
	print(response)
	print("if no error messege was printed just below dash line, succesfully disabled autofocus\n")

def enableDefaultAutofocus(camera):
	auto_focus = 1
	AUTO_FOCUS = AUTO_FOCUS_1 + str(camera) + AUTO_FOCUS_2 + str(auto_focus)
	response = os.popen(AUTO_FOCUS).read()
	print("------------------------------------------------------")
	print(response)
	print("if no error messege was printed just below dash line, succesfully enabled autofocus\n")

def set_focus(camera, focus_level):
	SET_FOCUS = SET_FOCUS_1 + str(camera) + SET_FOCUS_2 + str(focus_level)
	focus_resoponse = os.popen(SET_FOCUS).read()

def manualAutofucus(camera, cap, search_range=15):
	max_sharpness = 0
	global best_focus
	increment = 1
	count = 0
	current_focus = max(best_focus - search_range, 5)
	print("*-*-*-*-*-   Focusing Camera   -*-*-*-*")
	while(True):
		count = count + 1
		current_focus =  current_focus + increment
		set_focus(camera, current_focus)
		ret, original_frame = cap.read()
		gray_frame = cv2.cvtColor(np.uint8(original_frame), cv2.COLOR_BGR2GRAY)
		cv2.imshow("Fucusing", gray_frame)
		j = cv2.waitKey(300)
		if(j == 27):
			break
		sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
		print(current_focus, sharpness)
		if(sharpness > max_sharpness):
			max_sharpness = sharpness
			best_focus = current_focus
			continue
		# else:
		# 	current_focus = best_focus + search_range
		# 	increment = increment * -1
		if(count > search_range * 2):
			break
	set_focus(camera, best_focus)
	print("Focus point = " + str(best_focus))
	print("Max sharpness = " + str(max_sharpness))
	return max_sharpness

def configureCamera():
	camera = identifyCorrectCamera()
	disableDefaultAutofocus(camera)
	return True, camera


def endSession():
	print("Session terminated. Thank You!")
#****************************#

#************body************#

device_compatible = checkDevice()
camera_configured, camera = configureCamera()



while True:
    try:
        user = int(input("\nEnter User ID: "))
    except ValueError:
        print("Sorry, User ID should be an integer.")
        continue
    else:
    	print("\nUser ID: " + str(user) + "  has been registered:")
    	if(input("Do you wish to continue? (y/n): ") in ['y','Y']): 
        	break
    	else:
    		continue

if not(camera_configured and device_compatible):
	exit("Camera cinfuguration failed !!!")
    		
cap = cv2.VideoCapture(camera)
for x in range(1,2):
	manualAutofucus(camera, cap, search_range=15)
ret, frame = cap.read()
cv2.imshow("Fucus Check", frame)
in_key = cv2.waitKey()
while(chr(in_key) not in ['y','Y']):
	for x in range(1,3):
		manualAutofucus(camera, cap, search_range=15)
		ret, frame = cap.read()
		cv2.imshow("Fucus Check", frame)
		in_key = cv2.waitKey()
cv2.destroyAllWindows()


frame_count = len(os.listdir(DATAPATH))
while(True):
	ret, frame = cap.read()
	cv2.imshow("Eye", frame)
	in_key = cv2.waitKey(1)
	if(in_key == 27):
		break
	elif(in_key != -1):
		cv2.imwrite(DATAPATH + '/' + str(user) + '_' + str(frame_count) + '_' + chr(in_key) + '.jpg', frame)
		frame_count += 1
		print(frame_count)
#****************************#