from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch.utils.data as data
import os
import pickle
import pandas as pd
import torch
import random
import numpy as np
from torchvision import transforms
import cv2
import math



class CIFAR10(data.Dataset):

    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }


    def __init__(self, indexes, root: str, normal_class,
            task,
            download_data = False):
        super().__init__()

        self.task = task  # training set or test set
        self.data_path = root
        self.indexes = indexes
        self.normal_class = normal_class
        self.download_data = download_data

        if self.download_data:
            self.download()

        if (self.task == 'train') | (self.task == 'validate'):
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.data_path, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                if 'labels' in entry:
                    for i in range(0,len(entry['labels'])):
                        self.targets.extend([entry['labels'][i]])
                        self.data.append(entry['data'][i])

                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)

        self.targets = np.array(self.targets)
        if len(self.indexes) != 0: #if indexes is equal to [], original labels are not modified as this dataloader object is used by the 'create_reference' function. This function requires the original labels
          if (self.task == 'train') :
              self.data = np.array(self.data)[self.indexes]
              self.targets = self.targets[self.indexes]

          elif self.task == 'validate':
              lst = list(range(0,len(self.data) ))
              ind = [x for i,x in enumerate(lst) if (i not in self.indexes)]
              random.seed(1)
              val_indexes = random.sample(range(0, len(ind)), 1500)
              self.data = self.data[val_indexes]
              self.targets = self.targets[val_indexes]


          self.targets[self.targets != normal_class] = -1
          self.targets[self.targets == normal_class] = -2
          self.targets[self.targets == -2] = 0
          self.targets[self.targets == -1] = 1


    def download(self) -> None:
        download_and_extract_archive(self.url, self.data_path, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


    def __getitem__(self, index: int, seed = 1, base_ind=-1):

        img, target = self.data[index], int(self.targets[index])
        label = torch.Tensor([0])

        base=False
        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes.tolist()) ) #if img2 is the same as img, regenerate ind
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes.tolist()) )
                c = c+1

            if ind == base_ind:
              base = True

            img2, target2 = self.data[ind], int(self.targets[ind])

        else:
            img2 = torch.Tensor([1]) #not required if task is not equal to train
            label = torch.Tensor([target])

        img = torch.FloatTensor(img).squeeze(0).squeeze(0) #convert to 3x32x32
        img2 = torch.FloatTensor(img2).squeeze(0).squeeze(0)

        return img, img2, label, base



    def __len__(self) -> int:
        return len(self.data)
