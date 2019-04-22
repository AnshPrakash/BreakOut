import sys
import numpy as np
import pandas as pd
sys.path.append("./libsvm-3.23/python")
from svmutil import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import random
import pickle
from sklearn.decomposition import PCA
import os
import cv2





with open("pca.p",'rb') as f:
	pca = pickle.load(f)



code_dir = os.getcwd()
valid_set_dir = "./validation_dataset"


os.chdir(valid_set_dir)
with open("rewards.csv") as f:
	rewards = pd.read_csv(f,header = None)

labels = list(np.array(rewards.iloc[:,-1]))




folders = list(filter(os.path.isdir, os.listdir('.')))
folders.sort()

test  = []

for folder in folders:
	os.chdir(folder)
	print(folder)
	images = [(cv2.imread(str(i)+".png",cv2.IMREAD_GRAYSCALE)) for i in range(5)]
	img_array = np.array([x.ravel() for x in images])
	img_array = pca.transform(img_array)
	img_array =img_array.reshape(-1)
	test.append(img_array)
	os.chdir("..")
	
os.chdir(code_dir)
model= svm_load_model('libsvm_linear_kernel.model')
res = svm_predict(labels,test,model)
validation_accuracy = res[1][0]
print("validation accuracy using Linear Kernel",validation_accuracy)
print("Confusion matrix for Linear Kernel ")
print(confusion_matrix((labels),res[0]))
print("f1_score \n",f1_score(labels,res[0], average=None))
print("f1_score_macro \n",f1_score(labels,res[0], average='macro'))





model= svm_load_model('libsvm_Gaussian_kernel.model')
res = svm_predict(labels,test,model)
validation_accuracy = res[1][0]
print("validation accuracy using Gaussian Kernel",validation_accuracy)
print("Confusion matrix for Gaussian Kernel ")
print(confusion_matrix((labels),res[0]))
print("f1_score \n",f1_score(labels,res[0], average=None))
print("f1_score_macro \n",f1_score(labels,res[0], average='macro'))




