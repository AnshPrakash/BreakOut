import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

# os.chdir("00000001")
# files = list(filter(os.path.isfile, os.listdir('.')))
# files.sort()
# label_f = files[-1]
# del files[-1]
# for file in files:  
# 	img_array = cv2.imread(file,cv2.IMREAD_GRAYSCALE)
# 	img_array = cv2.resize(img_array, (210, 160))
# 	cv2.normalize(img_array, img_array, 0, 255, cv2.NORM_MINMAX)
# 	plt.imshow(img_array)
# 	mng = plt.get_current_fig_manager()
# 	mng.resize(*mng.window.maxsize())
# 	plt.pause(.001)
# 	plt.draw()
# print(img_array.shape)




# img_array = [cv2.imread("00000.png",cv2.IMREAD_GRAYSCALE),cv2.imread("00001.png",cv2.IMREAD_GRAYSCALE),cv2.imread("00002.png",cv2.IMREAD_GRAYSCALE),
# 			cv2.imread("00003.png",cv2.IMREAD_GRAYSCALE),cv2.imread("00004.png",cv2.IMREAD_GRAYSCALE)]
# temp = img_array[0]
# temp = temp[20:-15]
# for img in img_array[1:]:
# 	# temp = np.hstack((temp,img))
# 	temp = np.hstack((temp,img[20:-15]))


# plt.imshow(temp)
# plt.show()

img_array = cv2.imread("00000001_data.png",cv2.IMREAD_GRAYSCALE)
img_array = img_array[0]
img_array = img_array.reshape((875,160))

plt.imshow(img_array)
plt.show()

# img_array = cv2.resize(img_array, (210, 160))