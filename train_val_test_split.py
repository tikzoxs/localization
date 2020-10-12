import os
import random
import shutil

source_folder = "/home/1TB/new_iris_dataset/mixed_with_augmented"
train_folder = "/home/1TB/new_iris_dataset/train"
valaidation_folder = "/home/1TB/new_iris_dataset/validation"
test_folder = "/home/1TB/new_iris_dataset/test"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15

def allocateTo(target_folder, file):
	file_path = source_folder + '/' + file
	target_path = target_folder + '/' + file
	shutil.copyfile(file_path, target_path)

file_list = os.listdir(source_folder)
total_files = len(file_list)

val_amount = int(total_files * VAL_SPLIT)
test_amount = int(total_files * TEST_SPLIT)
train_amount = total_files - val_amount - test_amount

for file in file_list:
	print(train_amount, test_amount, val_amount)
	if(val_amount > 0 or test_amount > 0):
		folder_number = random.randint(1,100)
		if(folder_number % 3 == 0):
			val_test_selection = random.randint(1,2)
			if(val_amount > 0 and val_test_selection == 1):
				allocateTo(valaidation_folder, file)
				val_amount -= 1
			elif(test_amount > 0):
				allocateTo(test_folder, file)
				test_amount -= 1
		else:
			allocateTo(train_folder, file)
			train_amount -= 1
	else:
		allocateTo(train_folder, file)
		train_amount -= 1