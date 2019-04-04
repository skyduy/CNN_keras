import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import load_data

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # size: 3 * 150 * 40
        self.conv1 = nn.Conv2d(3, 32, 9)  # 32 * 142 * 32
        self.conv2 = nn.Conv2d(32, 32, 9)  # 32 * 134 * 24
        self.pool = nn.MaxPool2d(2)  # 32 * 67 * 12
        self.drop = nn.Dropout2d(0.25)
        self.fc = nn.Linear(32 * 67 * 12, 640 * 5)

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.drop4 = nn.Dropout(0.2)
        self.drop5 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(640 * 5, 33)
        self.fc2 = nn.Linear(640 * 5, 33)
        self.fc3 = nn.Linear(640 * 5, 33)
        self.fc4 = nn.Linear(640 * 5, 33)
        self.fc5 = nn.Linear(640 * 5, 33)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        x = x.view(-1, 32 * 67 * 12)
        x = self.fc(x)

        x1 = F.relu(self.fc1(self.drop1(x)))
        x2 = F.relu(self.fc2(self.drop2(x)))
        x3 = F.relu(self.fc3(self.drop3(x)))
        x4 = F.relu(self.fc4(self.drop4(x)))
        x5 = F.relu(self.fc5(self.drop5(x)))
        return [x1, x2, x3, x4, x5]

    def save(self, name, folder='./models'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        torch.save(self.state_dict(), path)

    def load(self, name, folder='./models'):
        path = os.path.join(folder, name)
        self.load_state_dict(torch.load(path))
        self.eval()


def validate(net, testloader, epoch):
    net.load(epoch)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data['image'], data['labels'].transpose(1, 0)
            outputs = net(inputs)  # batch
            for i in range(5):
                d = outputs[i].data
                _, res = torch.max(d, 1)
                ans = labels[i]
                total += labels.size(0)
                correct += (res == ans).sum().item()

    print('Accuracy of the network on {0:d} letters: {0:.2f} %%'
          .format(total, 100 * correct / total))


def train():
    trainloader, testloader, classes = load_data(batch_size=4)
    net = Net().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(net.parameters())
    for epoch in range(100):
        running_loss = 0.0
        for i_batch, data in enumerate(trainloader):
            inputs, labels = data['image'], data['labels'].transpose(1, 0)

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outs = net(inputs)
            loss = sum(criterion(outs[i], labels[i]) for i in range(5))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i_batch % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 10))
                running_loss = 0.0
        net.save(epoch)
        validate(net, testloader, epoch)

    net.save('finish')
    print('Finished Training')
