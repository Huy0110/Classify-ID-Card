import torch
import os
import matplotlib.pyplot as plt 
import torch.nn.functional as F 
import torch 
import numpy as np 
from torchvision import transforms as T,datasets
from torchvision.utils import save_image
from PIL import ImageFile
from PIL import Image, ImageEnhance, ImageOps
from config import CFG
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
ImageFile.LOAD_TRUNCATED_IMAGES = True

class Data_Loader():
    def __init__(self):
        #BaseDataLoader.initialize(self)
        self.train_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image to be 224 by 224
                             #T.RandomHorizontalFlip(),
                             T.RandomRotation(degrees=(-180,+180), expand = True), 
                             T.Resize(size=(CFG.img_size,CFG.img_size)), #Randomly Rotate Images by +/- 20 degrees, Image argumentation for each epoch
                             T.ColorJitter(brightness=0.6, contrast=0.4, saturation=0.4, hue=0.1),
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             #T.RandomErasing(p=0.5, scale = (0.05, 0.1), ratio =  (0.05, 0.1)),
                             T.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0)),
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                             ])
        self.validate_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image to be 224 by 224
                             #T.RandomRotation(degrees=(-20,+20)), #NO need for validation
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                             ])
        self.test_transform = T.Compose([
                             
                             T.Resize(size=(CFG.img_size,CFG.img_size)), # Resizing the image to be 224 by 224
                             #T.RandomRotation(degrees=(-20,+20)), #NO need for validation
                             T.ToTensor(), #converting the dimension from (height,weight,channel) to (channel,height,weight) convention of PyTorch
                             T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]) # Normalize by 3 means 3 StD's of the image net, 3 channels
                             ])
        
    def load_data(self):
        trainset=datasets.ImageFolder(CFG.train_path,transform=self.train_transform)
        print("Trainset Size:  {}".format(len(trainset)))
        validateset=datasets.ImageFolder(CFG.validate_path,transform=self.validate_transform)
        print("validateset Size:  {}".format(len(validateset)))
        testset=datasets.ImageFolder(CFG.test_path,transform=self.test_transform)
        print("testset Size:  {}".format(len(testset)))
        #class_name =["cc_2_front", "cc_back", "cm_back", "cm_front", "dl_back", "dl_front"]
        trainloader = DataLoader(trainset,batch_size=CFG.batch_size,shuffle=True)
        print("No. of batches in trainloader:{}".format(len(trainloader))) #Trainset Size:  5216 / batch_size: 16 = 326(No. of batches in trainloader) 
        print("No. of Total examples:{}".format(len(trainloader.dataset)))
        validationloader = DataLoader(validateset,batch_size=CFG.batch_size,shuffle=True)
        print("No. of batches in validationloader:{}".format(len(validationloader))) #validationset Size:  16 / batch_size: 16 = 1(No. of batches in validationloader) 
        print("No. of Total examples:{}".format(len(validationloader.dataset)))
        testloader = DataLoader(testset,batch_size=CFG.batch_size,shuffle=True)
        print("No. of batches in testloader:{}".format(len(testloader))) #testset Size:  624 / batch_size: 16 = 39(No. of batches in testloader) 
        print("No. of Total examples:{}".format(len(testloader.dataset)))
        return trainloader, validationloader, testloader