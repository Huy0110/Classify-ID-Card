import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
import torch.nn.functional as F 
import torch 
import numpy as np 
from torchvision import transforms as T,datasets
from torchvision.utils import save_image
from clearml import Task
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")
#task = Task.init(project_name='Card Classify', task_name='test model')
from PIL import ImageFile
from PIL import Image
from Model_Trainer import Card_Trainer
from config import CFG
from torch.utils.data import DataLoader
from torch import nn
import torch.nn.functional as F
import timm # PyTorch Image Models
ImageFile.LOAD_TRUNCATED_IMAGES = True
from data_loader import Data_Loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On which device we are on:{}".format(device))
data_loader = Data_Loader()
trainloader, validationloader, testloader = data_loader.load_data()



def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

def debug(list_labels, y_list_pred, list_images):
    os.makedirs('Debug_image', exist_ok = True)
    for i in range(list_labels.shape[0]):
        if list_labels[i]!=y_list_pred[i]:
            save_image(list_images[i], "Debug_image/" + target_names[list_labels[i]] + "_" + target_names[y_list_pred[i]] + ".jpg")

model = timm.create_model(CFG.model_name,pretrained=True) #load pretrained model

for param in model.parameters():
  param.requires_grad=False

#orginally, it was:
#(classifier): Linear(in_features=1792, out_features=1000, bias=True)

#we are updating it as a 2-class classifier:
model.classifier = nn.Sequential(
    nn.Linear(in_features=1792, out_features=625), #1792 is the orginal in_features
    nn.ReLU(), #ReLu to be the activation function
    nn.Dropout(p=0.3),
    nn.Linear(in_features=625, out_features=256),
    nn.ReLU(),
    nn.Linear(in_features=256, out_features=8), 
)

from torchsummary import  summary
model.to(device) # move the model to GPU
summary(model,input_size=(3,224,224))
target_names = ["ar", "cc_2_front", "cc_back", "cc_chip_back", "cc_chip_front", "cm_back", "cm_front", "dl_front"]

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = CFG.lr)

trainer = Card_Trainer(criterion,optimizer)
#trainer.fit(model,trainloader,validationloader,epochs = CFG.epochs)

model.load_state_dict(torch.load('model/model.pt'))
model.eval()

global_step = len(trainloader)

avg_test_loss, avg_test_acc = trainer.valid_batch_loop(model,testloader, global_step)


print("Test Loss : {}".format(avg_test_loss))
print("Test Acc : {}".format(avg_test_acc))

'''
import torch.nn.functional as F

testset_Internet =datasets.ImageFolder('Data',transform=test_transform)
class_name =["ar", "cc_2_front", "cc_back", "cc_chip_back", "cc_chip_front", "cm_back", "cm_front", "dl_front"]
testloader_internet = DataLoader(testset_Internet,batch_size=16,shuffle=True)

trainer.check(model,testloader_internet, 100)
#avg_test_loss, avg_test_acc = trainer.valid_batch_loop(model,testloader_internet, 100)
#print("Test Loss : {}".format(avg_test_loss))
#print("Test Acc : {}".format(avg_test_acc))
'''