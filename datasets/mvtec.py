import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import cv2
import random

class MVTEC(data.Dataset):


    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']



    def __init__(self, indexes, root: str, normal_class,
            task, data_path, seed,N,
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = data_path
        self.indexes = indexes
        self.normal_class = normal_class

        classes = ['bottle' , 'cable',  'capsule',  'carpet',  'grid',  'hazelnut',  'leather',  'metal_nut',  'pill',  'screw',  'tile',  'toothbrush',  'transistor',  'wood' , 'zipper']
        folder = classes[normal_class]

        self.data =[]
        self.targets=[]

        if self.task == 'train':
            path2 = root + folder + '/train/good/'
            images = os.listdir(path2)
            random.seed(seed)
            self.indexes = random.sample(list(range(0,len(images))), N)
            for ind in self.indexes:
                im = cv2.imread(path2 + images[ind])
                im2 =cv2.resize( im[:,:,0] , (64,64))
                im3 =cv2.resize( im[:,:,1] , (64,64))
                im4 =cv2.resize( im[:,:,2] , (64,64))
                im = np.stack((im2,im3,im4))
                self.data.append(im)

                self.targets.append(0)


        elif self.task == 'test':
            path2 = root + folder + '/test/'
            types = os.listdir(path2)
            for ty in types:
                path3 = path2 + ty
                images= os.listdir(path3)
                for image in images:
                    im = cv2.imread(path2 + ty + '/' + image)
                    im2 =cv2.resize( im[:,:,0] , (64,64))
                    im3 =cv2.resize( im[:,:,1] , (64,64))
                    im4 =cv2.resize( im[:,:,2] , (64,64))
                    im = np.stack((im2,im3,im4))
                    self.data.append(im)

                    if ty=='good':
                        self.targets.append(torch.Tensor([0]))
                    else:
                        self.targets.append(torch.Tensor([1]))

            print(len(self.data))
            print(len(self.targets))




    def __getitem__(self, index: int, seed = 1, base_ind=-1):



        base=False
        img, target = self.data[index], int(self.targets[index])
        img = torch.FloatTensor(img)

        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes) )
            c=1
            while (ind == index):
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
              base = True

            img2, target2 = self.data[ind], int(self.targets[ind])
            img2 = torch.FloatTensor(img2)
            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1])
            label = target



        return img, img2, label, base



    def __len__(self) -> int:
        return len(self.data)
