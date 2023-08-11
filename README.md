# FewSOME

This repository contains a Pytorch implementation of FewSOME: One-Class Few Shot Anomaly Detection with Siamese Networks.


## BibTeX Citation 

@inproceedings{belton2023fewsome,
  title={FewSOME: One-Class Few Shot Anomaly Detection With Siamese Networks},
  author={Belton, Niamh and Hagos, Misgina Tsighe and Lawlor, Aonghus and Curran, Kathleen M},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops},
  pages={2977--2986},
  year={2023}
}

## Abstract
Recent Anomaly Detection techniques have progressed the field considerably but at the cost of increasingly complex training pipelines. Such techniques require large amounts of training data, resulting in computationally expensive algorithms that are unsuitable for settings where only a small amount of normal samples are available for training. We propose '**Few** **S**hot an**OM**aly d**E**tection' (FewSOME), a deep One-Class Anomaly Detection algorithm with the ability to accurately detect anomalies having trained on \lq{}few\rq{} examples of the normal class and no examples of the anomalous class. We describe FewSOME to be of low complexity given its low data requirement and short training time. FewSOME is aided by pretrained weights with an architecture based on Siamese Networks. By means of an ablation study, we demonstrate how our proposed loss, 'Stop Loss', improves the robustness of FewSOME. Our experiments demonstrate that FewSOME performs at state-of-the-art level on benchmark datasets MNIST, CIFAR-10, F-MNIST and MVTec AD while training on only 30 normal samples, a minute fraction of the data that existing methods are trained on. Moreover, our experiments show FewSOME to be robust to contaminated datasets. We also report F1 score and balanced accuracy in addition to AUC as a benchmark for future techniques to be compared against. 


## Installation 
This code is written in Python 3.8 and requires the packages listed in requirements.txt.

Use the following command to clone the repository to your local machine:


```
git clone https://github.com/niamhbelton/FewSOME.git
```

To run the code, set up a virtual environment:

```
pip install virtualenv
cd <path-to-FewSOME-directory>
virtualenv myenv
source myenv/bin/activate
pip install -r requirements.txt
```


## Running Experiments


### MNIST Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-6 --batch_size 1 --weight_init_seed 1001 --dataset 'mnist' --normal_class 6 -N 30 --seed 1001 --eval_epoch 0 --epochs 6 --data_path ./data/ --download_data True --smart_samp 0 --k 1 --alpha 0.8 --vector_size 1024 --task test --pretrain 1 --model_type 'MNIST_VGG3'

```

| normal_class | epochs | alpha | batch_size | vector_size | k | lr | pretrain | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 8 | 0.01 | 1 | 1024 | 2 | 1e-5 | 1|
| 1 | 7 | 0.01 | 1 | 1024 | 2 | 1e-6 | 1|
| 2 | 6 | 0.01 | 1 | 1024 | 1 | 1e-5 | 1|
| 3 | 6 | 0.1 | 1 | 1024 | 1 | 1e-6 | 1|
| 4 | 5 | 0.01 | 1 | 2048 | 1 | 1e-5 | 1|
| 5 | 7 | 0.1 | 1 | 1024 | 1 | 1e-6 | 1|
| 6 | 6 | 0.8 | 1 | 1024 | 1 | 1e-6 | 1|
| 7 | 6 | 0.1 | 1 | 2048| 1 | 1e-5 | 1|
| 8 | 6 | 0.1 | 1 | 2048 | 3 | 1e-6 | 1|
| 9 | 6 | 0.01 | 1 | 1024 | 1 | 1e-5 | 1|



### CIFAR-10 Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-5 --batch_size 1 --weight_init_seed 1001 --dataset 'cifar10' --normal_class 0 -N 30 --seed 1001 --eval_epoch 0 --epochs 5 --data_path ./data/ --download_data True --smart_samp 0 --k 1 --alpha 0.5 --vector_size 2048 --task test --pretrain 1 --model_type 'CIFAR_VGG3'

```

| normal_class | epochs | alpha | batch_size | vector_size | k | lr | pretrain | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5 | 0.5 | 1 | 2048 | 1 | 1e-5 | 1|
| 1 | 11 | 0.1 | 8 | 2048 | 1 | 1e-4 | 1|
| 2 | 5 | 0.1 | 30 | 1024 | 3 | 1e-5 | 0 |
| 3 | 6 | 0.5 | 16 | 2048 | 3 | 1e-8 | 0 |
| 4 | 6 | 0.01 | 30 | 2048 | 1 | 1e-4 | 1|
| 5 | 6 | 0.1 | 1 | 2048 | 2 | 1e-5 | 1|
| 6 | 5 | 0.1 | 1 | 2048 | 2 | 1e-5 | 1|
| 7 | 6 | 0.01 | 1 | 1024| 2 | 1e-5 | 1|
| 8 | 6 | 0.5 | 1 | 2048 | 2 | 1e-5 | 1|
| 9 | 12 | 0.1 | 8 | 1024 | 1 | 1e-4 | 1|



### Fashion-MNIST Example

```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 30 --lr 1e-4 --batch_size 16 --weight_init_seed 1001 --dataset 'fashion' --normal_class 9 -N 30 --seed 1001 --eval_epoch 0 --epochs 6 --data_path ./data/ --download_data True --smart_samp 0 --k 1 --alpha 0.1 --vector_size 2048 --task test --pretrain 1 --model_type 'FASHION_VGG3'

```

| normal_class | epochs | alpha | batch_size | vector_size | k | lr | pretrain | 
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 4 | 0.01 | 16 | 1024 | 3 | 1e-8 | 1|
| 1 | 5 | 0.5 | 1 | 1024 | 1 | 1e-7 | 1|
| 2 | 4 | 0.1 | 16 | 1024 | 3 | 1e-8 | 1 |
| 3 | 5 | 0.2 | 16 | 1024 | 3 | 1e-5 | 1|
| 4 | 5 | 0.5 | 1 | 2048 | 3 | 1e-6 | 1|
| 5 | 6 | 1.0 | 30 | 2048 | 1 | 1e-7 | 1|
| 6 | 6 | 0.1 | 1 | 2048 | 3 | 1e-8 | 1|
| 7 | 5 | 0.2 | 8 | 2048| 2 | 1e-5 | 1|
| 8 | 6 | 0.1 | 30 | 2048 | 2 | 1e-8 | 1|
| 9 | 6 | 0.1 | 16 | 2048 | 1 | 1e-4 | 1|



### MVTec AD Example
This experiment requires the data to be downloaded from https://www.mvtec.com/company/research/datasets/mvtec-ad
```
cd <path-to-FewSOME-directory>

# activate virtual environment
source myenv/bin/activate

# change to source directory
cd src

# run experiment
python3 train.py -m model --num_ref_eval 60 --lr 1e-4 --batch_size 1 --weight_init_seed 1001 --dataset 'mvtec' --normal_class 13 -N 60 --seed 1001 --eval_epoch 1 --epochs 100 --data_path <path_to_data> --download_data True --smart_samp 0 --k 1 --alpha 1 --task test --pretrain 1 --model_type 'RESNET'

```

| normal_class | alpha | batch_size |  k | lr | pretrain | 
| --- | --- | --- | --- | --- | ---  |
| 0 | 1.0 | 16 | 1 | 1e-3 | 1|
| 1 | 0.5 | 30 | 1 | 1e-3 | 1|
| 2 | 0.01 | 16 | 1 | 1e-4 | 1|
| 3 | 1.0 | 16 | 1 | 1e-3 | 1|
| 4 | 0.01 | 8 | 1 | 1e-2 | 1|
| 5 | 0.8 | 30 | 1 | 1e-2 | 1|
| 6 | 0.5 | 1 | 1 | 1e-4 | 1|
| 7 | 0.8 | 16 | 1 | 1e-3 | 1|
| 8 | 0.01 | 30 | 1 | 1e-5 | 1|
| 9 | 0.8 | 1 | 1 | 1e-5 | 1|
| 10 | 1.0 | 1 | 1 | 1e-4 | 1|
| 11 | 0.01 | 16 | 1 | 1e-2 | 1|
| 12 | 0.8 | 16 | 1 | 1e-3 | 1|
| 13 | 1.0 | 1 | 1 | 1e-4 | 1|
| 14 | 0.8 | 16 | 1 | 1e-3 | 1|



## Desription of arguments for train.py

```
parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MVTEC_VGG3','MNIST_VGG3', 'MNIST_LENET', 'CIFAR_LENET', 'RESNET', 'FASHION_VGG3'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--smart_samp', type = int, choices = [0,1], default = 0)
    parser.add_argument('--k', type = int, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--num_ref_dist', type=int, default=None)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)

```

'-m' - this argument allows the user to enter a custom model name

'--model_type' - specify the architecture 

'--dataset' - specify the dataset 

'--normal_class' - specify the index of the class that is considered normal, all other classes are anomalies 

'-N' - specify the number of reference images 

'--num_ref_eval' - during training, the distance is calculated between the reference image in question and num_ref_eval reference images in order to find the reference images that are furthest away from the reference image in question. By default x=N. However, training can be sped up by setting x < N.

'--lr' - specify the learning rate

'--vector_size' - specify the number of elements in the 1D feature embedding (representation).

'--weight_decay' - specify the weight decay

'--seed' - specify the seed to select reference images

'--weight_init_seed' - specify the model seed

'--alpha' - specify the value of alpha between 0 and 1.

'--smart_samp' - specify whether to pair a reference image with reference images that have the largest euclidean distance.

'--k' - specify the number of reference images to calculate the distance from 

'--epochs' - specify the number of epochs 

'--data_path' - specify where to save the data

'--download_data' - specify whether to download data

'--contamination' - specify level of pollution i.e. the percentage of anomalies present in the training data 

'--v' - soft boundary parameter 

'--task' - value 'train' trains the model and tests on a validation set, 'test' trains the model and tests on the test set 

'--eval_epoch' - specifies whether to evaluate the model after each epoch 

'--pretrain' - specifies whether to use pretrained weights 

'--batch_size' - specify the batch size 

'--biases' - specify whether to turn off or on biases. 

'--early_stopping' - if the rate at which the loss is decreasing is less than .5% for a patience of 2, stop training.

'-i' - specify indexes of training set to have as a reference set 


## Output Files 
The code will create an 'outputs' folder. 
- For MNIST, Fashion MNIST and CIFAR-10, a directory for each class will be created (Example: 'outputs/class_0') that will contain all training details
- 'outputs/models/' - where each model is stored
- 'outputs/ED/' - where the Euclidean distance for each test data sample is stored. Each row is a test sample with columns 'label' (0 for anomaly and 1 for normal), 'minimum_dists' (the distance to nearest representation in the Reference Set), 'means' (the mean distance to each representation in the Reference set) and a column that shows the distance of the representation of the test sample to each representation in the Reference Set.
- 'outputs/losses/' - the training loss for each epoch
- outputs/inference_times/' - the inference times 


