import os
import torch
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms, functional

CHARS = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
         'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S',
         'T', 'U', 'V', 'W', 'X', 'Y', 'Z')

CAT2CHAR = dict(zip(range(len(CHARS)), CHARS))
CHAR2CAT = dict(zip(CHARS, range(len(CHARS))))


class ImageDataset(Dataset):
    def __init__(self, img_folder, train=True, transform=None, split_rate=0.2):
        wd, _ = os.path.split(os.path.abspath(__file__))
        self.folder = os.path.join(wd, img_folder)
        imgs = [i for i in os.listdir(self.folder) if i.endswith('jpg')]
        random.seed(1)
        random.shuffle(imgs)
        if not imgs:
            raise Exception('Empty folder!')

        point = int(split_rate * len(imgs))
        if train is True:
            self.imgs = imgs[point:]
        else:
            self.imgs = imgs[:point]
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        labels = [CHAR2CAT[self.imgs[idx][i]] for i in range(5)]
        path = os.path.join(self.folder, self.imgs[idx])
        im = Image.open(path)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        sample = {'image': im, 'labels': labels}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(transforms.ToTensor):
    def __call__(self, sample):
        im, labels = sample['image'], sample['labels']
        np_img = np.asarray(im)
        np_labels = np.asarray(labels)
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np_img.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float(),
                'labels': torch.from_numpy(np_labels).long()}


class Normalize(transforms.Normalize):
    def __call__(self, sample):
        tensor = sample['image']
        sample['image'] = functional.normalize(
            tensor, self.mean, self.std, self.inplace)
        return sample


def load_data(batch_size=4):
    transform = transforms.Compose([
        ToTensor(),
        Normalize([127.5, 127.5, 127.5], [128, 128, 128])
    ])

    trainset = ImageDataset('data', train=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size,
                             shuffle=True, num_workers=2)
    testset = ImageDataset('data', train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size,
                            num_workers=2)
    return trainloader, testloader, CHARS


def imshow(img):
    img = img * 128 + 127.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    im = Image.fromarray(npimg.astype('uint8'))
    im.show()


if __name__ == '__main__':
    import torchvision

    trains, tests, classes = load_data()
    dataiter = iter(trains)
    data = next(dataiter)
    images, labels = data['image'], data['labels']

    # show image and print labels
    imshow(torchvision.utils.make_grid(images))
    print(labels)
