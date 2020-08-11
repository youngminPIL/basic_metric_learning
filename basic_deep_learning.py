
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 24, 3, 1)
        self.conv2 = nn.Conv2d(24, 48, 3, 1)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(6912, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x


model = Net()
optimizer = optim.SGD(model.parameters(), 0.01)
criterion = nn.CrossEntropyLoss()


trn_dataset = datasets.MNIST('./mnist_data/',
                             download=True,
                             train=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(), # image to Tensor
                                 transforms.Normalize((0.1307,), (0.3081,)) # image, label
                             ]))

val_dataset = datasets.MNIST("./mnist_data/",
                             download=False,
                             train=False,
                             transform= transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307, ),(0.3081, ))
                             ]))


batch_size = 64
trn_loader = torch.utils.data.DataLoader(trn_dataset,
                                         batch_size=batch_size,
                                         shuffle=True)

val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)



model.train(True)
for epoch in range(1):
    trn_loss = 0.0
    for i, data in enumerate(tqdm(trn_loader, position=0, leave=True)):
        x, label = data
        # grad init
        optimizer.zero_grad()
        # forward propagation
        model_output = model(x)
        # calculate loss
        loss = criterion(model_output, label)
        trn_loss += loss
        # back propagation
        loss.backward()
        # weight update
        optimizer.step()


model.train(False)
val_loss = 0.0
val_acc = 0.0
for j, val in enumerate(tqdm(val_loader, position=0, leave=True)):
    val_x, val_label = val
    val_output = model(val_x)
    v_loss = criterion(val_output, val_label)
    _, preds = torch.max(val_output.data, 1)
    val_acc += torch.sum(preds == val_label.data)
    val_loss += v_loss

print(val_acc/10000)
