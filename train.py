import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from utils import load_data, DEVICE
from datetime import datetime


class Net(nn.Module):
    def __init__(self, gpu=False):
        super(Net, self).__init__()
        # size: 3 * 40 * 150
        self.conv1 = nn.Conv2d(3, 32, 9)  # 32 * 32 * 142
        self.conv2 = nn.Conv2d(32, 32, 9)  # 32 * 24 * 134
        self.pool = nn.MaxPool2d(2)  # 32 * 12 * 67
        self.drop1 = nn.Dropout2d(0.25)
        # flatten here
        self.fc1 = nn.Linear(32 * 12 * 67, 640 * 5)
        self.drop2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(640 * 5, 33 * 5)
        if gpu:
            self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.drop1(self.pool(x))
        x = x.view(-1, 32 * 12 * 67)
        x = self.drop2(self.fc1(x))
        x = self.fc2(x)
        return x

    def save(self, name, folder='./models'):
        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, name)
        torch.save(self.state_dict(), path)

    def load(self, name, folder='./models'):
        path = os.path.join(folder, name)
        self.load_state_dict(torch.load(path))
        self.eval()


def loss_batch(model, loss_func, data, opt=None):
    xb, yb = data['image'], data['label']
    out = model(xb)
    loss = loss_func(out, yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    loss_item = loss.item()
    del out
    del loss
    return loss_item, len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()  # train mode
        for i, data in enumerate(train_dl):
            loss_batch(model, loss_func, data, opt)

        model.eval()  # validate mode
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, data) for data in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        model.save('model-epoch-{}'.format(epoch + 1))
        print('Loss after epoch {}: {:.6f}'.format(epoch + 1, val_loss))


def train(use_gpu=True):
    train_dl, valid_dl = load_data(
        batch_size=4, max_m=4 * 9, split_rate=0.2, gpu=use_gpu)
    model = Net(use_gpu)
    opt = optim.Adadelta(model.parameters())
    criterion = nn.MultiLabelSoftMarginLoss()  # loss function
    fit(100, model, criterion, opt, train_dl, valid_dl)
    model.save('model-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
    print('Training finish')


if __name__ == '__main__':
    train()
