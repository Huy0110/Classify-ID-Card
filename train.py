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
task = Task.init(project_name='Card Classify', task_name='test model fix1')
from PIL import ImageFile
from PIL import Image, ImageEnhance, ImageOps
from config import CFG
from torch.utils.data import DataLoader
from data_loader import Data_Loader
from torch.utils.data import DataLoader
from Model_Trainer import Card_Trainer
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.makedirs('model', exist_ok = True)
continue_train = False


data_loader = Data_Loader()
trainloader, validationloader, testloader = data_loader.load_data()

from torch import nn
import torch.nn.functional as F
import timm # PyTorch Image Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On which device we are on:{}".format(device))

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

if continue_train == True:
  print("Load the epoch model")
  model.load_state_dict(torch.load('model_epoch.pt'))

summary(model,input_size=(3,224,224))
target_names = ["ar", "cc_2_front", "cc_back", "cc_chip_back", "cc_chip_front", "cm_back", "cm_front", "dl_front"]

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr = CFG.lr)

trainer = Card_Trainer(criterion,optimizer)
trainer.fit(model,trainloader,validationloader,epochs = CFG.epochs)