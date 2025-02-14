# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 17:13:12 2022

https://www.youtube.com/watch?v=oPhxf2fXHkQ&t=124s

https://towardsdatascience.com/building-rnn-lstm-and-gru-for-time-series-using-pytorch-a46e5b094e7b

https://github.com/jjdabr/forecastNet


https://www.deeplearningwizard.com/deep_learning/practical_pytorch/pytorch_feedforward_neuralnetwork/

https://www.tensorflow.org/tutorials/structured_data/time_series#performance

CUDA https://www.tensorflow.org/tutorials/structured_data/time_series#performance

https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html

@author: gabri
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# device confic - GPU - CUDA

device = torch.device ('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters

input_size = 784 # 28x28
hidden_size = 20
num_classes = 10
num_epochs = 2
batch_size = 10
learning_rate = 0.001

# MNIST

train_dataset = torchvision.datasets.MNIST (root='./data', train=True,
	transform=transforms.ToTensor(), download=True)

test_dataset = torchvision.datasets.MNIST (root='./data', train=False,
	transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
	shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size*2,
	shuffle=False)

#%%
# Show dataset examples

examples = iter(train_loader)

samples, labels = next(examples)
print(samples.shape, labels.shape)



for i in range(6):
 	plt.subplot(2,3,i+1)
 	plt.imshow(samples[i][0], cmap='gray')
plt.show()

#%%

# Define model

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
	
    def forward(self,x): 
        print(f'Step 1 - input: {x.shape}')
        out1 = self.l1(x)
        par1 = model.l1.weight
        print(f'Step 2 - l1: {out1.shape}')
        out2 = self.relu(out1)
        print(f'Step 3 - relu: {out2.shape}')
        out3 = self.l2(out2)
        par2 = model.l2.weight
        print(f'Step 4 - output: {out3.shape}')
        return out3, par1, par2

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# loss and optimized 

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%% training loop
model.train()
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
		# 100, 1, 28, 28
        # 100, 784
        images = images.reshape(-1,28*28).to(device) # data in 1 batch = 100 images x 784
        labels = labels.to(device)

        # forward
        outputs, par1, par2 = model(images) # feed the model only with 100 images
        
        for name, param in model.named_parameters():
            print(name)
            # print(param)
            
        loss = criterion(outputs, labels)

        # backwards
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f} ')



#%% test
model.eval()
with torch.no_grad():
	n_correct = 0
	n_samples = 0
	for images, labels in test_loader:
		images = images.reshape(-1, 28*28).to(device)	
		labels = labels.to(device)
		outputs = model(images)
	
		# value, index
		_, predictions = torch.max(outputs, 1)
		n_samples += labels.shape[0]
		n_correct += (predictions == labels).sum().item()


	acc = 100 * n_correct / n_samples
	print(f'accuracy = {acc}')

	