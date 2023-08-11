import torch.utils.data as data
from PIL import Image
from torchvision.datasets.utils import download_and_extract_archive, extract_archive, verify_str_arg, check_integrity
import torch
import random
import os
import codecs
import numpy as np
import random

class MNIST(data.Dataset):


    mirrors = [
        'http://yann.lecun.com/exdb/mnist/',
        'https://ossci-datasets.s3.amazonaws.com/mnist/',
    ]

    resources = [
        ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        ("t10k-images-idx3-ubyte.gz", "9fb629c4189551a2d022fa330f9573f3"),
        ("t10k-labels-idx1-ubyte.gz", "ec29112dd5afa0611ce80d1b7f02629c")
    ]


    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']



    def __init__(self, indexes, root: str, normal_class,
            task, data_path,
            download_data = False,
    ) -> None:
        super().__init__()
        self.task = task  # training set or test set
        self.data_path = data_path
        self.indexes = indexes
        self.normal_class = normal_class
        self.download_data = download_data



        if self.download_data:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        self.data, self.targets = self._load_data()

        if len(self.indexes) != 0: #if indexes is equal to [], original labels are not modified as this dataloader object is used by the 'create_reference' function. This function requires the original labels
          self.targets[self.targets != normal_class] = -1
          self.targets[self.targets == normal_class] = -2
          self.targets[self.targets == -2] = 0
          self.targets[self.targets == -1] = 1



    def get_int(self, b: bytes) -> int:
        return int(codecs.encode(b, 'hex'), 16)

    def read_sn3_pascalvincent_tensor(self, path: str, strict: bool = True) -> torch.Tensor:
        """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
           Argument may be a filename, compressed filename, or file object.
        """
        # read
        SN3_PASCALVINCENT_TYPEMAP = {
        8: (torch.uint8, np.uint8, np.uint8),
        9: (torch.int8, np.int8, np.int8),
        11: (torch.int16, np.dtype('>i2'), 'i2'),
        12: (torch.int32, np.dtype('>i4'), 'i4'),
        13: (torch.float32, np.dtype('>f4'), 'f4'),
        14: (torch.float64, np.dtype('>f8'), 'f8')
        }

        with open(path, "rb") as f:
            data = f.read()
        # parse
        magic = self.get_int(data[0:4])
        nd = magic % 256
        ty = magic // 256
        assert 1 <= nd <= 3
        assert 8 <= ty <= 14
        m = SN3_PASCALVINCENT_TYPEMAP[ty]
        s = [self.get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
        parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
        assert parsed.shape[0] == np.prod(s) or not strict
        return torch.from_numpy(parsed.astype(m[2])).view(*s)

    def read_image_file(self, path: str) -> torch.Tensor:
        x = self.read_sn3_pascalvincent_tensor(path, strict=False)
        assert(x.dtype == torch.uint8)
        assert(x.ndimension() == 3)
        return x

    def read_label_file(self, path: str) -> torch.Tensor:
        x = self.read_sn3_pascalvincent_tensor(path, strict=False)
        assert(x.dtype == torch.uint8)
        assert(x.ndimension() == 1)
        return x.long()




    def _load_data(self):
        if (self.task == 'train') | (self.task == 'validate'):
            image_file = "train-images-idx3-ubyte"
            data = self.read_image_file(os.path.join(self.data_path, image_file))
            label_file = "train-labels-idx1-ubyte"
            targets = self.read_label_file(os.path.join(self.data_path, label_file))

            if (self.task == 'train') & (len(self.indexes ) != 0):
                data = data[self.indexes]
                targets = targets[self.indexes]
            elif self.task == 'validate':
                lst = list(range(0,len(data) ))
                ind = [x for i,x in enumerate(lst) if i not in self.indexes]
                random.seed(1)
                randomlist = random.sample(range(0, len(ind)), 1500)
                data = data[randomlist]
                targets = targets[randomlist]
        else:
            image_file = "t10k-images-idx3-ubyte"
            data = self.read_image_file(os.path.join(self.data_path, image_file))
            label_file = "t10k-labels-idx1-ubyte"
            targets = self.read_label_file(os.path.join(self.data_path, label_file))

        return data, targets

    def __getitem__(self, index: int, seed = 1, base_ind=-1):



        base=False
        img, target = self.data[index], int(self.targets[index])
        img = torch.stack((img,img,img),0)

        if self.task == 'train':
            np.random.seed(seed)
            ind = np.random.randint(len(self.indexes) )
            c=1
            while (ind == index): #if img2 is the same as img, regenerate ind
                np.random.seed(seed * c)
                ind = np.random.randint(len(self.indexes) )
                c=c+1

            if ind == base_ind:
              base = True #img2 is equal to the anchor

            img2, target2 = self.data[ind], int(self.targets[ind])
            img2 = torch.stack((img2,img2,img2),0)
            label = torch.FloatTensor([0])
        else:
            img2 = torch.Tensor([1]) #if task is not equal to 'train', img2 is not required
            label = target



        return img, img2, label, base



    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def class_to_idx(self):
        return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self) -> bool:
        return all(
            check_integrity(os.path.join(self.data_path, os.path.splitext(os.path.basename(url))[0]))
            for url, _ in self.resources
        )

    def download(self) -> None:
        """Download the MNIST data if it doesn't exist already."""

        if self._check_exists():
            return

        os.makedirs('./data/', exist_ok=True)

        # download files
        for filename, md5 in self.resources:
            for mirror in self.mirrors:
                url = "{}{}".format(mirror, filename)
                try:
                    print("Downloading {}".format(url))
                    download_and_extract_archive(
                        url, download_root=self.data_path,
                        filename=filename,
                        md5=md5
                    )
                except URLError as error:
                    print(
                        "Failed to download (trying next):\n{}".format(error)
                    )
                    continue
                finally:
                    print()
                break
            else:
                raise RuntimeError("Error downloading {}".format(filename))

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.task == 0 else "Test")
