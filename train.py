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
        # size: 3 * 36 * 120
        self.conv1 = nn.Conv2d(3, 18, 5)  # 18 * 32 * 116
        self.pool1 = nn.MaxPool2d(2)  # 18 * 16 * 58
        self.drop1 = nn.Dropout(0.25)
        self.conv2 = nn.Conv2d(18, 48, 5)  # 48 * 12 * 54
        self.pool2 = nn.MaxPool2d(2)  # 48 * 6 * 27
        # flatten here
        self.drop2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(48 * 6 * 27, 1440)
        self.drop3 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(1440, 19 * 4)

        if gpu:
            self.to(DEVICE)

        if gpu:
            self.to(DEVICE)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 48 * 6 * 27)  # flatten here
        x = self.drop2(x)
        x = F.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x).view(-1, 4, 19)
        x = F.softmax(x, dim=2)
        x = x.view(-1, 4 * 19)
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
    batch_size = len(xb)
    out = model(xb)
    loss = loss_func(out, yb)

    single_correct, whole_correct = 0, 0
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
    else:  # calc accuracy
        yb = yb.view(-1, 4, 19)
        out_matrix = out.view(-1, 4, 19)
        _, ans = torch.max(yb, 2)
        _, predicted = torch.max(out_matrix, 2)
        compare = (predicted == ans)
        single_correct = compare.sum().item()
        for i in range(batch_size):
            if compare[i].sum().item() == 4:
                whole_correct += 1
        del out_matrix
    loss_item = loss.item()
    del out
    del loss
    return loss_item, single_correct, whole_correct, batch_size


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, verbose=None):
    for epoch in range(epochs):
        running_loss = 0.0
        model.train()  # train mode
        for i, data in enumerate(train_dl):
            loss, _, _, s = loss_batch(model, loss_func, data, opt)
            if isinstance(verbose, int):
                running_loss += loss * s
                if i % verbose == verbose - 1:
                    ave_loss = running_loss / (s * verbose)
                    print('[Epoch {}][Batch {}] got training loss: {:.6f}'
                          .format(epoch + 1, i + 1, ave_loss))
                    running_loss = 0.0

        model.eval()  # validate mode
        with torch.no_grad():
            losses, single, whole, batch_size = zip(
                *[loss_batch(model, loss_func, data) for data in valid_dl]
            )
        total_size = np.sum(batch_size)
        val_loss = np.sum(np.multiply(losses, batch_size)) / total_size
        single_rate = 100 * np.sum(single) / (total_size * 4)
        whole_rate = 100 * np.sum(whole) / total_size
        print('After epoch {}: \n'
              '\tLoss: {:.6f}\n'
              '\tSingle Acc: {:.2f}%\n'
              '\tWhole Acc: {:.2f}%'
              .format(epoch + 1, val_loss, single_rate, whole_rate))


def train(use_gpu=True):
    train_dl, valid_dl = load_data(
        batch_size=4, max_m=4 * 9, split_rate=0.2, gpu=use_gpu)
    model = Net(use_gpu)
    opt = optim.Adadelta(model.parameters())
    criterion = nn.BCELoss()  # loss function
    fit(100, model, criterion, opt, train_dl, valid_dl, 3)
    model.save('model-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S")))
    print('Training finish')


if __name__ == '__main__':
    train()
