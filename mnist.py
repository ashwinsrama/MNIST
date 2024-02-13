import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

def load_data(csv_file, transform=None):
    with open(csv_file, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)  # skip header
        
        images = []
        labels = []
        for row in csvreader:
            labels.append(int(row[0]))
            image = [float(pixel)/255.0 for pixel in row[1:]]
            if transform:
                image = transform(np.array(image)).tolist()
            images.append(image)
    return torch.tensor(images, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

print("...Generating Training Dataset...")
x_train, y_train = load_data('mnist/mnist_train.csv')

print("...Generating Testing Dataset...")
x_test, y_test = load_data('mnist/mnist_test.csv')

# Hyper-parameters
input_size = 784 # 28x28
hidden_size = 500 
num_classes = 10
batch_size = 64
num_epochs = 5
learning_rate = 0.001

train_dataset = TensorDataset(x_train, y_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Model definition
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

model = NeuralNet(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
print("...Training Model...")
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

# Testing the model
print("...Testing Model...")
model.eval()  # it will notify all your layers that you are in eval mode, that way, batchnorm or dropout layers will work in eval mode instead of training mode.
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct / total} %')
