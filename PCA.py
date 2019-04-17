import os
from sklearn.decomposition import PCA
from sklearn import svm
import numpy as np
import pandas as pd
import shutil
import cv2


os.chdir('RawData')
folders = list(filter(os.path.isdir, os.listdir('.')))
folders.sort()
currdir =os.getcwd()
l = []
img_array =[]

for folder in folders:
	os.chdir(folder)
	# files = list(filter(os.path.isfile, os.listdir('.')))
	# files.sort()
	# csv_file = files[-1]
	# del files[-1]
	imglist = [str(idx).zfill(5)+".png" for idx in range(0,50)]
	images = [cv2.imread(x,cv2.IMREAD_GRAYSCALE) for x in imglist]
	img_array = img_array + [x.ravel() for x in images]
	os.chdir(currdir)


# print(img_array)
train_data = np.array(img_array)

# train_data = train_data[:100]
print("Processing Done")


pca = PCA(n_components=50)
pca.fit(train_data)

# transformed = pca.transform(train_data)


os.chdir(currdir)
os.chdir("..")
try:
	os.mkdir("SVMData")
except Exception as e:
	print(e)

save_data =  os.getcwd()+"/SVMData"

print(save_data)
os.chdir(currdir)
for folder in folders:
	print(folder)
	os.chdir(folder)
	files = list(filter(os.path.isfile, os.listdir('.')))
	files.sort()
	csv_file = files[-1]
	images = [cv2.imread(x,cv2.IMREAD_GRAYSCALE) for x in files if x.endswith(".png")]
	img_array = np.array([x.ravel() for x in images])
	img_array = pca.transform(img_array)
	transformed_data = pd.DataFrame(img_array)
	transformed_data.to_csv(save_data +"/transformed_data_"+folder+".csv")
	src = os.getcwd()
	dst = save_data
	srcpath = os.path.join(src,csv_file)
	dstpath = os.path.join(dst, str(int(folder))+"_"+csv_file)
	shutil.copyfile(srcpath, dstpath)
	os.chdir(currdir)





