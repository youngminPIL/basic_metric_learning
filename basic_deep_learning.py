import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


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
        x = torch.flatten(x, 1)
        x = self.dropout(x)
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

test_dataset = datasets.MNIST("./mnist_data/",
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

test_loader = torch.utils.data.DataLoader(test_dataset,
                                         batch_size=batch_size,
                                         shuffle=False)


def train(train_iter):
    model.train(True) #training mode on
    trn_loss = 0.0
    for i, data in enumerate(tqdm(trn_loader, position=0, leave=True)):
        if i == train_iter:
            break
        x, label = data 
        optimizer.zero_grad() # grad init
        model_output = model(x) # forward propagation
        loss = criterion(model_output, label)  # calculate loss
        trn_loss += loss
        loss.backward() # back propagation
        optimizer.step()

def test():
    model.train(False) #training mode off
    test_acc = 0.0
    for _, test in enumerate(tqdm(test_loader, position=0, leave=True)):
        test_x, test_label =test
        test_output = model(test_x)
        _, preds = torch.max(test_output.data, 1)
        test_acc += torch.sum(preds == test_label.data)

    print('\nTest accuracy : {:.2f}'.format(test_acc/10000 *100))


def visualization(num):
    softmax = nn.Softmax(dim=0)
    get_score = model(val_loader.dataset[num][0].unsqueeze(axis=0))
    get_pred = torch.argmax(get_score)
    prob = softmax(get_score).squeeze()
    confidence_score = prob[get_pred]
    get_label = val_loader.dataset[num][1]

    fig = plt.figure()
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.set_title(
        'Prediction: %d, Confidence Score(%%): %.2f, Label: %d' % (get_pred, confidence_score * 100, get_label))

    subplot.imshow(np.array(val_loader.dataset[num][0]).reshape((28, 28)),
                   cmap=plt.cm.gray_r)
