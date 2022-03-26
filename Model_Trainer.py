from torchsummary import  summary
from torch import nn
import torch.nn.functional as F
import timm # PyTorch Image Models
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
#from clearml import Task
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")
#task = Task.init(project_name='Card Classify', task_name='test model fix1')
from PIL import ImageFile
from PIL import Image, ImageEnhance, ImageOps
from config import CFG
from torch.utils.data import DataLoader
from data_loader import Data_Loader
from torch.utils.data import DataLoader
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_names = ["ar", "cc_2_front", "cc_back", "cc_chip_back", "cc_chip_front", "cm_back", "cm_front", "dl_front"]

def accuracy(y_pred,y_true):
    y_pred = F.softmax(y_pred,dim = 1)
    top_p,top_class = y_pred.topk(1,dim = 1)
    equals = top_class == y_true.view(*top_class.shape)
    return torch.mean(equals.type(torch.FloatTensor))

class Card_Trainer():
    
    def __init__(self,criterion = None,optimizer = None,schedular = None):
        
        self.criterion = criterion
        self.optimizer = optimizer
        self.schedular = schedular
    
    def train_batch_loop(self,model,trainloader,global_step):
        
        train_loss = 0.0
        train_acc = 0.0
        running_loss = 0.0
        count =0
        
        for images,labels in tqdm(trainloader): 
            
            # move the data to CPU
            #print(labels)
            images = images.to(device)
            labels = labels.to(device)
            
            logits = model(images)
            #print(logits.size)
            #print(labels.size)
            loss = self.criterion(logits,labels)
            
            self.optimizer.zero_grad()
            running_loss +=loss.item()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_acc += accuracy(logits,labels)
            global_step+=1
            count+=1
            if global_step % 50 == 0:
                writer.add_scalar('train_loss', running_loss / 50, global_step)
                running_loss = 0.0
        return train_loss / len(trainloader), train_acc / len(trainloader) , global_step

    
    def valid_batch_loop(self,model,validloader, global_step):
        
        valid_loss = 0.0
        valid_acc = 0.0
        val_loss = 0.0
        count = 0
        for images,labels in tqdm(validloader):
            
            # move the data to CPU
            images = images.to(device) 
            labels = labels.to(device)
            
            logits = model(images)
            loss = self.criterion(logits,labels)
            
            valid_loss += loss.item()
            valid_acc += accuracy(logits,labels)
            
            count+=1
            y_pred = F.softmax(logits,dim = 1)
            top_p,y_pred = y_pred.topk(1,dim = 1)
            y_pred = y_pred.view(*labels.shape)
            if count == 1:
                y_list_pred = y_pred.detach().cpu().numpy()
                list_labels = labels.detach().cpu().numpy()
            else:
                y_list_pred = np.concatenate([y_list_pred, y_pred.detach().cpu().numpy()])
                list_labels = np.concatenate([list_labels, labels.detach().cpu().numpy()])
            #print(labels.detach().cpu().numpy())
            #print(y_pred.detach().cpu().numpy())
            #print(classification_report(labels.detach().cpu().numpy(), y_pred.detach().cpu().numpy(), target_names=target_names))

        writer.add_scalar('val_loss', valid_loss, global_step)
        print(classification_report(list_labels, y_list_pred, target_names=target_names))
        return valid_loss / len(validloader), valid_acc / len(validloader)
            
        
    def fit(self,model,trainloader,validloader,epochs):
        
        valid_min_loss = np.Inf 
        global_step = 0
        
        for i in range(epochs):
            
            model.train() # this turn on dropout
            avg_train_loss, avg_train_acc, global_step = self.train_batch_loop(model,trainloader, global_step) ###
            
            model.eval()  # this turns off the dropout lapyer and batch norm
            avg_valid_loss, avg_valid_acc = self.valid_batch_loop(model,validloader, global_step) ###
            
            if avg_valid_loss <= valid_min_loss :
                print("Valid_loss decreased {} --> {}".format(valid_min_loss,avg_valid_loss))
                torch.save(model.state_dict(),'model/model.pt')
                valid_min_loss = avg_valid_loss
            
            torch.save(model.state_dict(), 'model/model_epoch.pt')
            print("Save the epoch model")

                
            print("Epoch : {} Train Loss : {:.6f} Train Acc : {:.6f}".format(i+1, avg_train_loss, avg_train_acc))
            print("Epoch : {} Valid Loss : {:.6f} Valid Acc : {:.6f}".format(i+1, avg_valid_loss, avg_valid_acc))
    def check(self,model,validloader, global_step):
        
        valid_loss = 0.0
        valid_acc = 0.0
        val_loss = 0.0
        count = 0
        count_batch = 0
        for images,labels in tqdm(validloader):
            

            # move the data to CPU
            images = images.to(device)
            labels = labels.to(device)

            count+=1
            list_labels_2 = labels.detach().cpu().numpy()
            for i in range(list_labels_2.shape[0]):
                save_image(images[i],"debug2/" + target_names[list_labels_2[i]]  + str(count) + "_" + str(i) + ".jpg")
            
            if count == 10:
                break