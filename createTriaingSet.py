import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib
import random

def selectSample(l,n):
	random.shuffle(l)
	b = l[:n]
	b.sort()
	return(b)


def imageStacked(imglist):
	img_array = [cv2.imread(x,cv2.IMREAD_GRAYSCALE) for x in imglist]
	temp = img_array[0]
	temp = temp[20:-15]
	for img in img_array[1:]:
		# tp =img[20:-15]
		# print(tp.shape)
		temp = np.vstack((temp,img[20:-15]))
	return(temp.ravel())

def getRewardedSample(indices,df):
	samples = []
	idxStart = indices - 7 +1 #all are one shifted
	for idx in idxStart:
		# l = [i +idx for i in range(7)]
		# l = selectSample(l,5)
		l = [i +idx for i in range(6)]
		l = selectSample(l,4)
		l.append(idx + 6)
		img =[str(s).zfill(5)+".png" for s in l]
		samples.append((imageStacked(img),1.0))
	return(samples)



def getRandomSamples(df):
	samples = []
	start = int(np.random.rand()*10) + 1
	stride = 0
	while start < (len(df)-8):
		# l = [1,2,3,4,5,6]
		# img = [str(start+s).zfill(5)+".png" for s in ([0] + selectSample(l,4))] 
		l = [0,1,2,3,4,5]
		img = [str(start+s).zfill(5)+".png" for s in (selectSample(l,4) + [6])] 
		temp = imageStacked(img)
		samples.append((temp,df.iloc[start + 6,-1]))
		stride = random.randint(20,30)
		start += stride
	return(samples)




try:
	os.mkdir("TrainingData")
except Exception as e:
	print(e)


os.chdir('RawData')
folders = list(filter(os.path.isdir, os.listdir('.')))
folders.sort()
currdir =os.getcwd()
l = [] 
for folder in folders:
	os.chdir(folder)
	files = list(filter(os.path.isfile, os.listdir('.')))
	files.sort()
	csv_file = files[-1]
	del files[-1]
	with open(csv_file) as f:
		df = pd.read_csv(f,header = None)
		# print(len(df))
		df.index = np.arange(1,len(df)+1)
	# print(df)
	indxOfRewards = df.index[df[0]== 1] - 1 
	samples = getRewardedSample(indxOfRewards,df)
	samples = samples + (getRandomSamples(df))
	random.shuffle(samples)
	os.chdir("../..")
	os.chdir("TrainingData")
	data = [0]*len(samples)
	labels  = [0]*len(samples)
	for i in range(len(samples)):
		data[i]   = (samples[i][0])
		labels[i] = (samples[i][1])
	del samples
	matplotlib.image.imsave(folder + '_data.png', np.array(data))
	labels_df = pd.DataFrame(np.array(labels))
	del data
	del labels
	labels_df.to_csv(folder+"_labels.csv")
	print(len(labels_df))
	del labels_df
	os.chdir(currdir)

print("Processing Done")