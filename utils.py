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

ONE_HOT = torch.eye(len(CHARS))


class ImageDataset(Dataset):
    def __init__(self, folder, img_list, transform=None):
        self.folder = folder
        self.im_list = img_list
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        label = self.im_list[idx][:5]
        path = os.path.join(self.folder, self.im_list[idx])
        im = Image.open(path)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        sample = {'image': im, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Word2OneHot(object):
    def __call__(self, sample):
        labels = list()
        for c in sample['label']:
            idx = CHARS.index(c)
            labels.append(ONE_HOT[idx])
        sample['label'] = torch.cat(labels)
        return sample


class ImgToTensor(object):
    def __call__(self, sample):
        np_img = np.asarray(sample['image'])
        image = np_img.transpose((2, 0, 1))  # H x W x C  -->  C x H x W
        sample['image'] = torch.from_numpy(image).float()
        return sample


class Normalize(transforms.Normalize):
    def __call__(self, sample):
        tensor = sample['image']
        sample['image'] = functional.normalize(
            tensor, self.mean, self.std, self.inplace)
        return sample


class ToGPU(object):
    def __call__(self, sample):
        device = torch.device("cuda:0")
        sample['image'] = sample['image'].to(device)
        sample['label'] = sample['label'].long().to(device)
        return sample


def load_data(batch_size=4, max_m=-1, split_rate=0.2, gpu=False):
    # list images
    wd, _ = os.path.split(os.path.abspath(__file__))
    folder = os.path.join(wd, 'data')
    imgs = [i for i in os.listdir(folder) if i.endswith('jpg')]
    if not imgs:
        raise Exception('Empty folder!')
    random.seed(1)
    random.shuffle(imgs)
    point = int(split_rate * len(imgs))
    train_imgs = imgs[point:][:max_m]
    valid_imgs = imgs[:point][:max_m]

    # initialize transform
    chains = [Word2OneHot(),
              ImgToTensor(),
              Normalize([127.5, 127.5, 127.5], [128, 128, 128])]
    if gpu and torch.cuda.is_available():
        chains.append(ToGPU())
    transform = transforms.Compose(chains)

    # load data
    train_ds = ImageDataset(folder, train_imgs, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True, num_workers=2)
    valid_ds = ImageDataset(folder, valid_imgs, transform=transform)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size,
                          num_workers=2)
    return train_dl, valid_dl


def imshow(img):
    img = img * 128 + 127.5  # unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    im = Image.fromarray(npimg.astype('uint8'))
    im.show()


if __name__ == '__main__':
    import torchvision

    trains, tests = load_data()
    dataiter = iter(trains)
    data = next(dataiter)
    images, cats = data['image'], data['labels']

    # show image and print labels
    imshow(torchvision.utils.make_grid(images))
    print(cats)
