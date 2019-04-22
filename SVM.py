import sys
import numpy as np
import pandas as pd
# sys.path.append("/home/cse/btech/cs1160367/home/ML_Assign4/libsvm-3.23/python")
sys.path.append("./libsvm-3.23/python")
from svmutil import *
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import random

train_f = "./SVMInput/FINAL_data.csv"
labels_f = "./SVMInput/FINAL_labels.csv"

# train_f = "/home/cse/btech/cs1160367/home/ML_Assign4/SVMInput/FINAL_data.csv"
# labels_f = "/home/cse/btech/cs1160367/home/ML_Assign4/SVMInput/FINAL_labels.csv"


with open(train_f) as f:
	train_data = pd.read_csv(f)

train_data = np.array(train_data.iloc[:,1:])

with open(labels_f) as f:
	labels = pd.read_csv(f)

labels = np.array(labels.iloc[:,1:])
print("loading complete")




# labels =labels[:10]
# train_data = train_data[:10]


Y_p = list(labels.reshape(-1))
X_p = list(train_data)

samples = random.sample(list(zip(X_p,Y_p)),10000)

X_p =[]
Y_p =[]
for x,y in samples:
	X_p.append(x)
	Y_p.append(y)

print("Sampling Complete")


def training_Linear_models(C):
	global X_p,Y_p
	param  = svm_parameter('-s 0 -t 0 -c '+ str(C))
	prob   = svm_problem(Y_p,X_p)
	model  = svm_train(prob,param)
	return(model)




C = 1.0
model = training_Linear_models(C)
svm_save_model('libsvm_linear_kernel.model', model)
print("Training Complete for Linear Kernel")
res = svm_predict(Y_p,X_p,model)
training_accuracy = res[1][0]
print("Training accuracy using Linear Kernel",training_accuracy)
print("Confusion matrix for Linear Kernel ")
print(confusion_matrix(Y_p,res[0]))
print("f1_score \n",f1_score(Y_p,res[0], average=None))
print("f1_score_macro \n",f1_score(Y_p,res[0], average='macro'))





def training_Gaussian_models(G):
	global X_p,Y_p
	C = 1.0
	param  = svm_parameter('-s 0 -t 2 -c '+ str(C) +' -g '+str(G))
	prob   = svm_problem(Y_p,X_p)
	model  = svm_train(prob,param)
	return(model)



G = 0.05
model_g = training_Gaussian_models(C)
svm_save_model('libsvm_Gaussian_kernel.model', model_g)
print("Training Complete for Gaussian Kernel")
res = svm_predict(Y_p,X_p,model_g)
training_accuracy = res[1][0]
print("Training accuracy using Linear Kernel",training_accuracy)
print("Confusion matrix for Linear Kernel ")
print(confusion_matrix(Y_p,res[0]))
print("f1_score \n",f1_score(Y_p,res[0], average=None))
print("f1_score_macro \n",f1_score(Y_p,res[0], average='macro'))




linearModel = svm_load_model('libsvm_linear_kernel.model')
Gaussian_Model = svm_load_model('libsvm_Gaussian_kernel.model')



valid_set_dir = "./validation_dataset"
os.chdir(valid_set_dir)
curr_dir = os.getcwd()
with open("rewards.csv") as f:
	rewards = pd.read_csv(f,header = None)

labels = np.array(rewards.iloc[:,-1])


