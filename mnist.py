import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import random
import numpy as np
from random import sample
from numpy import asarray
from PIL import Image

#get data
def get_array(src):
    img = Image.open(src)
    data = []
    for row in asarray(img):
        data += list(row)
    return data

###############################################################################################
###############################################################################################
######################################## TESTING DATA #########################################
###############################################################################################
###############################################################################################

print("...Generating Training Dataset...")
train_batch = []
for i in range(10):
    train_batch += sample([('MNIST/training/' + str(i) + '/' + file, i) for file in os.listdir('MNIST/training/' + str(i))], 1500)

random.shuffle(train_batch)

x_train = []
y_train = []
for src, label in train_batch:
    x_train += [get_array(src)]
    y_train.append(label)

y_train = torch.Tensor(y_train).type(torch.LongTensor)
x_train = torch.Tensor(x_train)

print("done\n")


###############################################################################################
###############################################################################################
######################################## TRAINING DATA ########################################
###############################################################################################
###############################################################################################

print("...Generating Testing Dataset...")
test_batch = []
for i in range(10):
    test_batch += sample([('MNIST/testing/' + str(i) + '/' + file, i) for file in os.listdir('MNIST/testing/' + str(i))], 500)

random.shuffle(test_batch)

x_test = []
y_test = []
for src, label in test_batch:
    x_test += [get_array(src)]
    y_test.append(label)

y_test = torch.Tensor(y_test).type(torch.LongTensor)
x_test = torch.Tensor(x_test)

print("done\n")

###############################################################################################
###############################################################################################
############################################# MODEL ###########################################
###############################################################################################
###############################################################################################

print("...Creating Model...")
# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
num_iters = 250
learning_rate = 0.001


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.sigmoid2 = nn.Sigmoid()
        self.l3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.l1(x)
        out = self.sigmoid1(out)
        out = self.l2(out)
        out = self.sigmoid2(out)
        out = self.l3(out)
        return out

#Initialize model with loss and optimizer
model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)  

print("done\n")

###############################################################################################
###############################################################################################
############################################ TRAINING #########################################
###############################################################################################
###############################################################################################

print("...Training Model...")
# Train the model
for i in range(num_iters):
    # Forward pass
    y_train_pred = model(x_train)
    loss = criterion(y_train_pred, y_train)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print Training Loss
    if (i+1) % 10 == 0:
        print (f'Iteration [{i+1}/{num_iters}], Loss: {loss.item():.6f}')
print("done\n")


###############################################################################################
###############################################################################################
############################################# TESTING #########################################
###############################################################################################
###############################################################################################


print("...Testing Model...")
#Test the model
num_correct = 0
num_samples = len(y_test)

res = model(x_test)
_, y_test_pred = torch.max(res.data, 1)

for i in range(num_samples):
    if y_test_pred[i] == y_test[i]:
        num_correct += 1

acc = 100.0 * num_correct / num_samples
print(f'Accuracy of the network on the 10000 test images: {acc} %\n')
