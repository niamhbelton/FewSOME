import torch.nn.functional as F
import torch
from model import *
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_curve
from sklearn import metrics
from datasets.main import load_dataset
import random
from sklearn.metrics import f1_score
import time


def deactivate_batchnorm(m):
    '''
        Deactivate batch normalisation layers
    '''
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, alpha, anchor, device, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()

        self.margin = margin
        self.v = v
        self.alpha = alpha
        self.anchor = anchor
        self.device = device

    def forward(self, output1, vectors, label):
        '''
        Args:
            output1 - feature embedding/representation of current training instance
            vectors - list of feature embeddings/representations of training instances to contrast with output1
            label - value of zero if output1 and all vectors are normal, one if vectors are anomalies
        '''

        euclidean_distance = torch.FloatTensor([0]).to(self.device)

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)/ torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device))


        euclidean_distance += self.alpha*((F.pairwise_distance(output1, self.anchor)) /torch.sqrt(torch.Tensor([output1.size()[1]])).to(self.device) )

        #calculate the margin
        marg = (len(vectors) + self.alpha) * self.margin

        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss

        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        return loss_contrastive


def evaluate(anchor, seed, base_ind, ref_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, data_path, criterion, alpha, num_ref_eval, device):

    model.eval()


    #create loader for test dataset
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, num_ref_eval))
    np.random.shuffle(ind)
    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    for i in ind:
      img1, _, _, _ = ref_dataset.__getitem__(i)
      if (i == base_ind):
        ref_images['images{}'.format(i)] = anchor
      else:
        ref_images['images{}'.format(i)] = model.forward( img1.to(device).float())

      outs['outputs{}'.format(i)] =[]




    means = []
    minimum_dists=[]
    lst=[]
    labels=[]
    loss_sum =0
    inf_times=[]
    total_times= []
    #loop through images in the dataloader

    with torch.no_grad():
        for i, data in enumerate(loader):

            image = data[0][0]
            label = data[2].item()

            labels.append(label)
            total =0
            mini=torch.Tensor([1e50])
            t1 = time.time()
            out = model.forward(image.to(device).float()) #get feature vector (representation) for test image
            inf_times.append(time.time() - t1)

            #calculate the distance from the test image to each of the datapoints in the reference set
            for j in range(0, num_ref_eval):
                euclidean_distance = (F.pairwise_distance(out, ref_images['images{}'.format(j)]) / torch.sqrt(torch.Tensor([out.size()[1]])).to(device) ) + (alpha*(F.pairwise_distance(out, anchor) /torch.sqrt(torch.Tensor([out.size()[1]])).to(device) ))

                outs['outputs{}'.format(j)].append(euclidean_distance.item())
                total += euclidean_distance.item()
                if euclidean_distance.detach().item() < mini:
                  mini = euclidean_distance.item()

                loss_sum += criterion(out,[ref_images['images{}'.format(j)]], label).item()

            minimum_dists.append(mini)
            means.append(total/len(indexes))
            total_times.append(time.time()-t1)

            del image
            del out
            del euclidean_distance
            del total
            torch.cuda.empty_cache()


    #create dataframe of distances to each feature vector in the reference set for each test feature vector
    cols = ['label','minimum_dists', 'means']
    df = pd.concat([pd.DataFrame(labels, columns = ['label']), pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)
    for i in range(0, num_ref_eval):
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))
    df.columns=cols
    df = df.sort_values(by='minimum_dists', ascending = False).reset_index(drop=True)


    #calculate metrics
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['minimum_dists']))
    auc_min = metrics.auc(fpr, tpr)
    outputs = np.array(df['minimum_dists'])
    thres = np.percentile(outputs, 10)
    outputs[outputs > thres] =1
    outputs[outputs <= thres] =0
    f1 = f1_score(np.array(df['label']),outputs)
    fp = len(df.loc[(outputs == 1 ) & (df['label'] == 0)])
    tn = len(df.loc[(outputs== 0) & (df['label'] == 0)])
    fn = len(df.loc[(outputs == 0) & (df['label'] == 1)])
    tp = len(df.loc[(outputs == 1) & (df['label'] == 1)])
    spec = tn / (fp + tn)
    recall = tp / (tp+fn)
    acc = (recall + spec) / 2
    print('AUC: {}'.format(auc_min))
    print('F1: {}'.format(f1))
    print('Balanced accuracy: {}'.format(acc))
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['means']))
    auc = metrics.auc(fpr, tpr)



    #create dataframe of feature vectors for each image in the reference set
    feat_vecs = pd.DataFrame(ref_images['images0'].detach().cpu().numpy())
    for j in range(1, num_ref_eval):
        feat_vecs = pd.concat([feat_vecs, pd.DataFrame(ref_images['images{}'.format(j)].detach().cpu().numpy())], axis =0)

    avg_loss = (loss_sum / num_ref_eval )/ val_dataset.__len__()

    return auc, avg_loss, auc_min, f1,acc, df, feat_vecs, inf_times, total_times




def init_feat_vec(model,base_ind, train_dataset, device ):
        '''
        Initialise the anchor
        Args:
            model object
            base_ind - index of training data to convert to the anchor
            train_dataset - train dataset object
            device
        '''

        model.eval()
        anchor,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          anchor = model(anchor.to(device).float())

        return anchor



def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    '''
    Get indexes for reference set
    Include anomalies in the reference set if contamination > 0
    Args:
        contamination - level of contamination of anomlies in reference set
        dataset name
        normal class
        task - train/test/validate
        data_path - path to data
        download data
        N - number in reference set
        seed
    '''
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class,task, data_path, download_data) #get all training data
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0] #get indexes in the training set that are equal to the normal class
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N) #randomly sample N normal data points
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.ceil(N*contamination)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0] #get indexes of non-normal class
      samp = random.sample(range(0, len(con)), int(numb))
      samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
      final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MNIST_VGG3', 'RESNET'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--num_ref_dist', type=int, default = None)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    num_ref_eval = args.num_ref_eval
    N = args.num_ref
    if num_ref_eval == None:
        num_ref_eval = N


    #if indexes for reference set aren't provided, create the reference set.
    if args.dataset != 'mvtec':
        if args.index != []:
            indexes = [int(item) for item in indexes.split(', ')]
        else:
            indexes = create_reference(args.contamination, args.dataset, args.normal_class, 'train', args.data_path, args.download_data, N, args.seed)


    #set the seed
    torch.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed_all(args.weight_init_seed)


    #Initialise the model
    if args.model_type == 'CIFAR_VGG3':
        if args.pretrain == 1:
            model = CIFAR_VGG3_pre(args.vector_size, args.biases)
        else:
            model = CIFAR_VGG3(args.vector_size, args.biases)
    elif args.model_type == 'MNIST_VGG3':
        if args.pretrain == 1:
            model = MNIST_VGG3_pre(args.vector_size, args.biases)
        else:
            model = MNIST_VGG3(args.vector_size, args.biases)
    elif args.model_type == 'RESNET':
        model = RESNET_pre()
    elif (args.model_type == 'FASHION_VGG3'):
        if (args.pretrain ==1):
            model = FASHION_VGG3_pre(args.vector_size, args.biases)
        else:
            model = FASHION_VGG3(args.vector_size, args.biases)


    if (args.model_type == 'RESNET'):
        model.apply(deactivate_batchnorm)

    #create datasets
    if args.dataset == 'mvtec':
        ref_dataset = load_dataset(args.dataset, args.index, args.normal_class, 'train', args.data_path, download_data=True, seed=args.seed, N=N)
        indexes = ref_dataset.indexes
        val_dataset = load_dataset(args.dataset, args.index, args.normal_class, 'test', args.data_path, download_data=True, seed=args.seed, N=N)
    else:
        ref_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'train', args.data_path, download_data=True)
        val_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'test', args.data_path, download_data=True)

    #initialise the anchor
    model.to(args.device)
    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    np.random.seed(args.epochs)
    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]
    anchor = init_feat_vec(model,base_ind , ref_dataset, args.device)

    #load model
    model.load_state_dict(torch.load(args.model_path + args.model_name))

    criterion = ContrastiveLoss(args.alpha, anchor, args.device, args.v)

    val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs, inf_times, total_times = evaluate(anchor, args.seed, base_ind, ref_dataset, val_dataset, model, args.dataset, args.normal_class, args.model_name, indexes, args.data_path, criterion, args.alpha, args.num_ref_eval, args.device)

    #write out all details of model training
    cols = ['normal_class', 'auc_min','f1','acc']
    params = [args.normal_class, val_auc_min, f1,acc]
    string = './outputs/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(args.normal_class)+'/'+args.model_name)
