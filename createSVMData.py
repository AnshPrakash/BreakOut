import os
import numpy as np
import pandas as pd
import random


def selectSample(l,n):
	random.shuffle(l)
	b = l[:n]
	b.sort()
	return(b)



def getRewardedSample(indices,data):
	samples = []
	idxStart = indices - 7
	for idx in idxStart:
		l = [i +idx for i in range(6)]
		l = selectSample(l,4)
		l.append(idx + 6)
		l = np.array(l) 
		fseq = np.array(data.iloc[l+1]).reshape(-1)
		samples.append((fseq,1.0))
	return(samples)



def getRandomSamples(labels,data):
	samples = []
	start = int(np.random.rand()*10) + 1
	stride = 0
	while start < (len(labels)-8):
		l = [0,1,2,3,4,5]
		l = selectSample(l,4)+[6]
		temp = np.array(data.iloc[np.array(l)+1]).reshape(-1)
		samples.append((temp,labels.iloc[start + 6,-1]))
		stride = random.randint(20,30)
		start += stride
	return(samples)


# os.chdir("/home/cse/btech/cs1160367/home/ML_Assign4") ##for hpc

try:
	os.mkdir("SVMInput")
except Exception as e:
	print(e)


os.chdir('PCA_Out')

samples = [] 
for i in range(1,501):
	print("Processing Game",i)
	label_file = str(i)+"_rew.csv"
	data_file = "transformed_data_" + str(i).zfill(8) + ".csv"
	with open(label_file) as f:
		labels = pd.read_csv(f,header = None)
	with open(data_file) as f:
		data = pd.read_csv(f)
		data = data.iloc[:,1:]
	indxOfRewards = labels.index[labels[0]== 1] # We have rename the Indices
	samples = samples + getRewardedSample(indxOfRewards,data)
	samples = samples + (getRandomSamples(labels,data))

os.chdir("../SVMInput")
random.shuffle(samples)

final_data = [0]*len(samples)
final_labels  = [0]*len(samples)
for i in range(len(samples)):
	final_data[i]   = (samples[i][0])
	final_labels[i] = (samples[i][1])

del samples
final_data = np.array(final_data)
final_labels = np.array(final_labels)

data_df = pd.DataFrame(final_data)
labels_df = pd.DataFrame(final_labels)
print("Writing to Disk ... ")
data_df.to_csv("FINAL_data.csv")
labels_df.to_csv("FINAL_labels.csv")

print("Processing Done SVM Data Done")