import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt
from skimage import io, transform
from torch.autograd import Variable
import cv2
import os


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
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.conv2 = nn.Conv2d(in_channels = 32,out_channels = 64, kernel_size = 3,stride =2,padding =(0,0))
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride = 2)
        self.fc = nn.Linear(37440,2048)
        self.out_layer = nn.Linear(2048, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc(x))
        x = self.out_layer(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:       # Get the products
            num_features *= s
        return num_features


net = Net()
criterion = nn.NLLLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)
epochs =1

for _ in range(epochs):
    # for i in range(1,501):
    for i in range(1,2):
        TrainingDataset = BreakOutDataset(file='TrainingData/'+str(i).zfill(8))
        dataloader = DataLoader(TrainingDataset, batch_size=4,shuffle=False, num_workers=4)
        for i_batch, sample_batched in enumerate(dataloader):
            inputs = Variable(sample_batched['image'])
            target = Variable(sample_batched['label'])
            optimizer.zero_grad()   # zero the gradient buffers
            output = net(inputs)
            print(output,target)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()    # Does the update

# with torch.no_grad():
#         for param in model.parameters():
#             param -= learning_rate * param.grad
#             