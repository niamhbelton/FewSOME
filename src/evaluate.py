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

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, v=0.0,margin=0.8):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.v = v

    def forward(self, output1, vectors, feat1, label, alpha):
        euclidean_distance = torch.FloatTensor([0]).cuda()

        #get the euclidean distance between output1 and all other vectors
        for i in vectors:
          euclidean_distance += (F.pairwise_distance(output1, i)/ torch.sqrt(torch.Tensor([output1.size()[1]])).cuda())


        euclidean_distance += alpha*((F.pairwise_distance(output1, feat1)) /torch.sqrt(torch.Tensor([output1.size()[1]])).cuda() )

        #calculate the margin
        marg = (len(vectors) + alpha) * self.margin


        #if v > 0.0, implement soft-boundary
        if self.v > 0.0:
            euclidean_distance = (1/self.v) * euclidean_distance
        #calculate the loss

        loss_contrastive = ((1-label) * torch.pow(euclidean_distance, 2) * 0.5) + ( (label) * torch.pow(torch.max(torch.Tensor([ torch.tensor(0), marg - euclidean_distance])), 2) * 0.5)

        return loss_contrastive




def evaluate(feat1, seed, base_ind, ref_dataset, val_dataset, model, dataset_name, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval):

    model.eval()


    #create loader for dataset for test set
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False)
    outs={} #a dictionary where the key is the reference image and the values is a list of the distances between the reference image and all images in the test set
    ref_images={} #dictionary for feature vectors of reference set
    ind = list(range(0, num_ref_eval))
    np.random.shuffle(ind)
    #loop through the reference images and 1) get the reference image from the dataloader, 2) get the feature vector for the reference image and 3) initialise the values of the 'out' dictionary as a list.
    for i in ind:
      img1, _, _, _ = ref_dataset.__getitem__(i)
      if (i == base_ind):
        ref_images['images{}'.format(i)] = feat1
      else:
        ref_images['images{}'.format(i)] = model.forward( img1.cuda().float())

      outs['outputs{}'.format(i)] =[]




    means = []
    minimum_dists=[]
    lst=[]
    labels=[]
    loss_sum =0
    inf_times=[]
    total_times= []
    #loop through images in the dataloader
    for i, data in enumerate(loader):

        if i % 1000 == 0:
          print(i)

        image = data[0][0]
        label = data[2].item()

        labels.append(label)
        total =0
        mini=torch.Tensor([1e50])
        t1 = time.time()
        out = model.forward(image.cuda().float()) #get feature vector for test image
        inf_times.append(time.time() - t1)
        #calculate the distance from the test image to each of the datapoints in the reference set


        for j in range(0, num_ref_eval):
            euclidean_distance = (F.pairwise_distance(out, ref_images['images{}'.format(j)]) / torch.sqrt(torch.Tensor([out.size()[1]])).cuda() ) + (alpha*(F.pairwise_distance(out, feat1) /torch.sqrt(torch.Tensor([out.size()[1]])).cuda() ))

            outs['outputs{}'.format(j)].append(euclidean_distance.item())
            total += euclidean_distance.item()
            if euclidean_distance.detach().item() < mini:
              mini = euclidean_distance.item()

            loss_sum += criterion(out,[ref_images['images{}'.format(j)]], feat1,label, alpha).item()

        minimum_dists.append(mini)
        means.append(total/len(indexes))

        total_times.append(time.time()-t1)



        del image
        del out
        del euclidean_distance
        del total
        torch.cuda.empty_cache()



    cols = ['label','minimum_dists', 'means']
    df = pd.concat([pd.DataFrame(labels, columns = ['label']), pd.DataFrame(minimum_dists, columns = ['minimum_dists']),  pd.DataFrame(means, columns = ['means'])], axis =1)

    for i in range(0, num_ref_eval):
        df= pd.concat([df, pd.DataFrame(outs['outputs{}'.format(i)])], axis =1)
        cols.append('ref{}'.format(i))

    df.columns=cols
    df = df.sort_values(by='minimum_dists', ascending = False).reset_index(drop=True)



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

    anoms = len(df.loc[(df['label'] == 1) & (df['minimum_dists'] > thres)])
    normals = len(df.loc[(df['label'] == 0) & (df['minimum_dists'] <= thres)])
    print('AUC based on minimum vector {}'.format(auc_min))
    print('F1 based on minimum vector {}'.format(f1))
    print('Balanced accuracy based on minimum vector {}'.format(acc))
    fpr, tpr, thresholds = roc_curve(np.array(df['label']),np.array(df['means']))
    auc = metrics.auc(fpr, tpr)

    feat_vecs = pd.DataFrame(ref_images['images0'].detach().cpu().numpy())
    for j in range(1, num_ref_eval):
        feat_vecs = pd.concat([feat_vecs, pd.DataFrame(ref_images['images{}'.format(j)].detach().cpu().numpy())], axis =0)

    avg_loss = (loss_sum / num_ref_eval )/ val_dataset.__len__()

    name = model_name + '_' + '_num_ref_eval_' + str(num_ref_eval) + '_seed_' + str(seed) + '_normal_class_' + str(normal_class) + '_auc_' + str(auc_min)
    pd.DataFrame([np.mean(inf_times), np.std(inf_times), np.mean(total_times), np.std(total_times), auc_min ,f1,acc]).to_csv('./outputs/inference_times/class_'+str(normal_class)+'/'+name)


    return auc, avg_loss, auc_min, f1,acc, df, feat_vecs




def init_feat_vec(model,base_ind, train_dataset ):

        model.eval()
        feat1,_,_,_ = train_dataset.__getitem__(base_ind)
        with torch.no_grad():
          feat1 = model(feat1.cuda().float()).cuda()

        return feat1


def create_reference(contamination, dataset_name, normal_class, task, data_path, download_data, N, seed):
    indexes = []
    train_dataset = load_dataset(dataset_name, indexes, normal_class, task,  data_path, download_data)
    ind = np.where(np.array(train_dataset.targets)==normal_class)[0]
    random.seed(seed)
    samp = random.sample(range(0, len(ind)), N)
    final_indexes = ind[samp]
    if contamination != 0:
      numb = np.ceil(N*contamination)
      if numb == 0.0:
        numb=1.0

      con = np.where(np.array(train_dataset.targets)!=normal_class)[0]
      samp = random.sample(range(0, len(con)), int(numb))
      samp2 = random.sample(range(0, len(final_indexes)), len(final_indexes) - int(numb))
      final_indexes = np.array(list(final_indexes[samp2]) + list(con[samp]))
    return final_indexes



def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MNIST_VGG3', 'MNIST_LENET', 'CIFAR_LENET', 'RESNET'], required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--output_name', type=str, required=True)
    parser.add_argument('--normal_class', type=int, default = 0)
    parser.add_argument('-N', '--num_ref', type=int, default = 30)
    parser.add_argument('--num_ref_eval', type=int, default = None)
    parser.add_argument('--vector_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default = 100)
    parser.add_argument('--weight_init_seed', type=int, default = 100)
    parser.add_argument('--alpha', type=float, default = 0)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--data_path',  required=True)
    parser.add_argument('--download_data',  default=True)
    parser.add_argument('--contamination',  type=float, default=0)
    parser.add_argument('--v',  type=float, default=0.0)
    parser.add_argument('--task',  default='train', choices = ['test', 'train'])
    parser.add_argument('--eval_epoch', type=int, default=0)
    parser.add_argument('--biases', type=int, default=1)
    parser.add_argument('--pretrain', type=int, default=1)
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    model_name = args.model_name
    model_path = args.model_path
    model_type = args.model_type
    dataset = args.dataset
    output_name = args.output_name
    normal_class = args.normal_class
    N = args.num_ref
    seed = args.seed
    epochs = args.epochs
    data_path = args.data_path
    download_data = args.download_data
    contamination = args.contamination
    indexes = args.index
    alpha = args.alpha
    vector_size = args.vector_size
    weight_init_seed = args.weight_init_seed
    v = args.v
    task = args.task
    eval_epoch = args.eval_epoch
    biases = args.biases
    num_ref_eval = args.num_ref_eval
    pretrain = args.pretrain
    if num_ref_eval == None:
        num_ref_eval = N


    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/inference_times'):
        os.makedirs('outputs/inference_times')
    if not os.path.exists('outputs/inference_times/class_' + str(normal_class)):
        os.makedirs('outputs/inference_times/class_'+str(normal_class))


    #if indexes for reference set aren't provided, create the reference set.
    if indexes != []:
        indexes = [int(item) for item in indexes.split(', ')]
    else:
        indexes = create_reference(contamination, dataset, normal_class, 'train', data_path, download_data, N, seed)



    #set the seed
    torch.manual_seed(weight_init_seed)
    torch.cuda.manual_seed(weight_init_seed)
    torch.cuda.manual_seed_all(weight_init_seed)


    #Initialise the model
    if model_type == 'CIFAR_VGG3':
        if args.pretrain == 1:
            model = CIFAR_VGG3_pre(vector_size, biases)
        else:
            model = CIFAR_VGG3(vector_size, biases)
    elif model_type == 'MNIST_VGG3':
        if args.pretrain == 1:
            model = MNIST_VGG3_pre(vector_size, biases)
        else:
            model = MNIST_VGG3(vector_size, biases)
    elif model_type == 'MNIST_LENET':
        model = MNIST_LeNet(vector_size, biases)
    elif model_type == 'RESNET':
        model = RESNET_pre(vector_size, biases)
    elif model_type == 'CIFAR_LENET':
        model = CIFAR_LeNet(vector_size, biases)
    elif (model_type == 'CIFAR_VGG4'):
        if (args.pretrain ==1):
            model = CIFAR_VGG4_pre(vector_size, biases)
        else:
            model = CIFAR_VGG4(vector_size, biases)


    if (model_type == 'RESNET'):
        model.apply(deactivate_batchnorm)

    ref_dataset = load_dataset(dataset, indexes, normal_class, 'train', data_path, download_data=True)
    val_dataset = load_dataset(dataset, indexes, normal_class, 'test', data_path, download_data=True)


    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]
    feat1 = init_feat_vec(model.cuda(),base_ind, ref_dataset )


    model.load_state_dict(torch.load(model_path + model_name))
    model.cuda()

    criterion = ContrastiveLoss(v)





    val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs = evaluate(feat1, seed, base_ind, ref_dataset, val_dataset, model, dataset, normal_class, output_name, model_name, indexes, data_path, criterion, alpha, num_ref_eval)


    print('AUC is {}'.format(val_auc_min))
    print('F1 is {}'.format(f1))
    print('Balanced accuracy is {}'.format(acc))
