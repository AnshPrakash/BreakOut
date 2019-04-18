import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.autograd import Variable
import numpy as np
import cv2
import os


# Check if gpu support is available
cuda_avail = torch.cuda.is_available()

class BreakOutDataset(Dataset):
    """BreakOut dataset."""

    def __init__(self, file, transform=None):
        self.images = cv2.imread(file+"_data.png",cv2.IMREAD_GRAYSCALE)
        self.labels = pd.read_csv(file+"_labels.csv")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_label = self.labels.iloc[idx,-1]
        image = self.images[idx]
        image = image.reshape(3,1050,160)
        image = torch.from_numpy(image).float()
        img_label = img_label.astype(int)
        # image = image.reshape(1050,160,3)
        sample = {'image': image, 'label': img_label}
        if self.transform:
            sample = self.transform(sample)
        return sample


# TrainingDataset = BreakOutDataset(file='TrainingData/'+str(1).zfill(8))

# fig = plt.figure()
# for i in range(len(TrainingDataset)):
#     sample = TrainingDataset[i]

#     print(i, sample['image'].shape, sample['label'])

#     ax = plt.subplot(1, 4, i + 1)
#     plt.tight_layout()
#     ax.set_title('Sample #{}'.format(i))
#     ax.axis('off')
#     plt.imshow(sample['image'])
#     # show_landmarks(**sample)
#     if i == 3:
#         # plt.show()
#         break

# dataloader = DataLoader(TrainingDataset, batch_size=4,
#                         shuffle=False, num_workers=4)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch, sample_batched['image'].size(),sample_batched['label'])


def save_models(epoch):
    torch.save(model.state_dict(), "BreakOutmodel_{}.model".format(epoch))
    print("Chekcpoint saved")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3,out_channels = 32, kernel_size = 3,stride =2,padding =(0,0))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3,stride =2,padding =(0,0))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc = nn.Linear(37440,2048)
        self.out_layer = nn.Linear(2048, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        # print("After Conv1",x)
        x = self.bn1(x)
        x = self.pool1(x)
        # print("After Max pool1",x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # print("After conv 2",x)
        x = self.pool2(x)
        # print("After Max pool 2",x)
        x = x.view(-1, self.num_flat_features(x))
        # print(x)
        x = F.relu(self.fc(x))
        # print("After Hideen Layer",x)
        x = self.out_layer(x)
        # print("After Output layer",x)
        x = F.softmax(x,dim=1)
        # print(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features

def accuracy(file):
    accuracy = 0
    total =0
    TrainingDataset = BreakOutDataset(file=file)
    dataloader = DataLoader(TrainingDataset, batch_size=4,shuffle=False, num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        if cuda_avail:
            train_data = (sample_batched['image'].cuda())
            target = (sample_batched['label'].cuda())
        else:                
            train_data = (sample_batched['image'])
            target = (sample_batched['label'])
        with torch.no_grad():
            pred = net(train_data)
            total +=(len(target))
            # print(len(target))
            maxs, argmax = torch.max(pred, 1)
            accuracy += torch.sum(torch.eq(argmax,target)).float()
    return(accuracy/total)



net = Net().cuda() if cuda_avail else Net()
print("Cuda ",cuda_avail)


criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.00001,momentum = 0.9,weight_decay=1e-5)
epochs =1
batch_size = 4

for _ in range(epochs):
    for i in range(1,501):
        TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
        dataloader = DataLoader(TrainingDataset, batch_size=batch_size,shuffle=False, num_workers=4)
        for i_batch, sample_batched in enumerate(dataloader):
            if cuda_avail:
                inputs = Variable(sample_batched['image'].cuda())
                target = Variable(sample_batched['label'].cuda())
            else:                
                inputs = Variable(sample_batched['image'])
                target = Variable(sample_batched['label'])
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(inputs)
            loss = criterion(output.cpu(), target.cpu())
            loss.backward()
            optimizer.step()    # Does the update
            # print(torch.max(output,1)[-1],target)
            # print("probabilties ",output)
            # print(str(loss) +"\n")
        print("Accuracy ",accuracy('TrainingData/'+str(i).zfill(8)))

