import torch
from datasets.main import load_dataset
from model import *
import os
import numpy as np
import pandas as pd
import argparse
import torch.nn.functional as F
import torch.optim as optim
from evaluate import evaluate
import random
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



def create_batches(lst, n):
    '''
    Args:
        lst - list of indexes for training instances
        n - batch size
    '''
    for i in range(0, len(lst), n):
        yield lst[i:i + n]



def train(model, lr, weight_decay, train_dataset, val_dataset, epochs, criterion, alpha, model_name, indexes, data_path, normal_class, dataset_name, smart_samp, k, eval_epoch, model_type, bs, num_ref_eval, num_ref_dist, device):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses = []
    patience = 0
    max_patience = 2 #patience based on train loss
    max_iter = 0
    patience2 = 10 #patience based on evaluation AUC
    best_val_auc = 0
    best_val_auc_min = 0
    best_f1=0
    best_acc=0
    stop_training = False


    start_time = time.time()

    for epoch in range(epochs):
        print("Starting epoch " + str(epoch+1))

        model.train()

        loss_sum = 0

        #create batches for epoch
        np.random.seed(epoch)
        np.random.shuffle(ind)
        batches = list(create_batches(ind, bs))

        #iterate through each batch
        for i in range(int(np.ceil(len(ind) / bs))):

            #iterate through each training instance in batch
            for batch_ind,index in enumerate(batches[i]):

                seed = (epoch+1) * (i+1) * (batch_ind+1)
                img1, img2, labels, base = train_dataset.__getitem__(index, seed, base_ind)

                # Forward
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)

                if (index ==base_ind):
                  output1 = anchor
                else:
                  output1 = model.forward(img1.float())

                if (smart_samp == 0) & (k>1):

                  vecs=[]
                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, k):
                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vecs.append(output2)

                elif smart_samp == 0:

                  if (base == True):
                    output2 = anchor
                  else:
                    output2 = model.forward(img2.float())

                  vecs = [output2]

                else:
                  max_eds = [0] * k
                  max_inds = [-1] * k
                  max_ind =-1
                  vectors=[]

                  ind2 = ind.copy()
                  np.random.seed(seed)
                  np.random.shuffle(ind2)
                  for j in range(0, num_ref_dist):

                    if ((num_ref_dist ==1) & (ind2[j] == base_ind)) | ((num_ref_dist ==1) & (ind2[j] == index)):
                        c = 0
                        while ((ind2[j] == base_ind) | (index == ind2[j])):
                            np.random.seed(seed * c)
                            j = np.random.randint(len(ind) )
                            c = c+1

                    if (ind2[j] != base_ind) & (index != ind2[j]):
                      output2=model(train_dataset.__getitem__(ind2[j], seed, base_ind)[0].to(device).float())
                      vectors.append(output2)
                      euclidean_distance = F.pairwise_distance(output1, output2)

                      for b, vec in enumerate(max_eds):
                          if euclidean_distance > vec:
                            max_eds.insert(b, euclidean_distance)
                            max_inds.insert(b, len(vectors)-1)
                            if len(max_eds) > k:
                              max_eds.pop()
                              max_inds.pop()
                            break

                  vecs = []

                  for x in max_inds:
                      with torch.no_grad():
                          vecs.append(vectors[x])

                if batch_ind ==0:
                    loss = criterion(output1,vecs,labels)
                else:
                    loss = loss + criterion(output1,vecs,labels)

            loss_sum+= loss.item()
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()


        train_losses.append((loss_sum / len(ind))) #average loss for each training instance

        print("Epoch: {}, Train loss: {}".format(epoch+1, train_losses[-1]))


        if (eval_epoch == 1):
            training_time = time.time() - start_time
            eval_start_time = time.time()
            val_auc, val_loss, val_auc_min, f1, acc,df, ref_vecs, inf_times, total_times = evaluate(anchor, seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, data_path, criterion, alpha, num_ref_eval, device)
            eval_time = time.time() - eval_start_time
            print('Validation AUC is {}'.format(val_auc))
            print("Epoch: {}, Validation loss: {}".format(epoch+1, val_loss))
            if val_auc_min > best_val_auc_min:
                best_val_auc = val_auc
                best_val_auc_min = val_auc_min
                best_epoch = epoch
                best_f1 = f1
                best_acc = acc
                best_df=df
                max_iter = 0
                training_time_best = (time.time() - start_time) - (eval_time*(epoch+1))

                training_time = (time.time() - start_time) - (eval_time*(epoch+1))
                write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time_best,f1,acc, train_losses, inf_times, total_times)

            else:
                max_iter+=1

            if max_iter == patience2:
                break

        elif args.early_stopping ==1:
            if epoch > 1:
              decrease = (((train_losses[-3] - train_losses[-2]) / train_losses[-3]) * 100) - (((train_losses[-2] - train_losses[-1]) / train_losses[-2]) * 100)

              if decrease <= 0.5:
                patience += 1


              if (patience==max_patience) | (epoch == epochs-1):
                  stop_training = True


        elif (epoch == (epochs -1)) & (eval_epoch == 0):
            stop_training = True




        if stop_training == True:
            print("--- %s seconds ---" % (time.time() - start_time))
            training_time = time.time() - start_time
            val_auc, val_loss, val_auc_min, f1,acc, df, ref_vecs, inf_times, total_times = evaluate(anchor,seed, base_ind, train_dataset, val_dataset, model, dataset_name, normal_class, model_name, indexes, data_path, criterion, alpha, num_ref_eval, device)


            write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times)

            break



    print("Finished Training")
    if eval_epoch == 1:
        print("AUC was {} on epoch {}".format(best_val_auc_min, best_epoch+1))
        return best_val_auc, best_epoch, best_val_auc_min, training_time_best, best_f1, best_acc,train_losses
    else:
        print("AUC was {} on epoch {}".format(val_auc_min, epoch+1))
        return val_auc, epoch, val_auc_min, training_time, f1,acc, train_losses




def write_results(model_name, normal_class, model, df, ref_vecs,num_ref_eval, num_ref_dist, val_auc, epoch, val_auc_min, training_time,f1,acc, train_losses, inf_times, total_times):
    '''
        Write out results to output directories and save model
    '''

    model_name_temp = model_name + '_epoch_' + str(epoch+1) + '_val_auc_' + str(np.round(val_auc, 3)) + '_min_auc_' + str(np.round(val_auc_min, 3))
    for f in os.listdir('./outputs/models/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/models/class_'+str(normal_class) + '/{}'.format(f))
    torch.save(model.state_dict(), './outputs/models/class_'+str(normal_class)+'/' + model_name_temp)


    for f in os.listdir('./outputs/ED/class_'+str(normal_class) + '/'):
      if (model_name in f) :
          os.remove(f'./outputs/ED/class_'+str(normal_class) + '/{}'.format(f))
    df.to_csv('./outputs/ED/class_'+str(normal_class)+'/' +model_name_temp)

    for f in os.listdir('./outputs/ref_vec/class_'+str(normal_class) + '/'):
      if (model_name in f) :
        os.remove(f'./outputs/ref_vec/class_'+str(normal_class) + '/{}'.format(f))
    ref_vecs.to_csv('./outputs/ref_vec/class_'+str(normal_class) + '/' +model_name_temp)


    pd.DataFrame([np.mean(inf_times), np.std(inf_times), np.mean(total_times), np.std(total_times), val_auc_min ,f1,acc]).to_csv('./outputs/inference_times/class_'+str(normal_class)+'/'+model_name_temp)

     #write out all details of model training
    cols = ['normal_class', 'ref_seed', 'weight_seed', 'num_ref_eval', 'num_ref_dist','alpha', 'lr', 'weight_decay', 'vector_size','biases', 'smart_samp', 'k', 'v', 'contam' , 'AUC', 'epoch', 'auc_min','training_time', 'f1','acc']
    params = [normal_class, args.seed, args.weight_init_seed, num_ref_eval, num_ref_dist, args.alpha, args.lr, args.weight_decay, args.vector_size, args.biases, args.smart_samp, args.k, args.v, args.contamination, val_auc, epoch+1, val_auc_min, training_time,f1,acc]
    string = './outputs/class_' + str(normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    pd.DataFrame([params], columns = cols).to_csv('./outputs/class_'+str(normal_class)+'/'+model_name)
    pd.DataFrame(train_losses).to_csv('./outputs/losses/class_'+str(normal_class)+'/'+model_name)


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
    parser.add_argument('--model_type', choices = ['CIFAR_VGG3','CIFAR_VGG4','MVTEC_VGG3','MNIST_VGG3', 'RESNET', 'FASHION_VGG3'], required=True)
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
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('-i', '--index', help='string with indices separated with comma and whitespace', type=str, default = [], required=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':

    args = parse_arguments()
    N = args.num_ref
    num_ref_eval = args.num_ref_eval
    num_ref_dist = args.num_ref_dist
    if num_ref_eval == None:
        num_ref_eval = N
    if num_ref_dist == None:
        num_ref_dist = N

    #if indexes for reference set aren't provided, create the reference set.
    if args.dataset != 'mvtec':
        if args.index != []:
            indexes = [int(item) for item in indexes.split(', ')]
        else:
            indexes = create_reference(args.contamination, args.dataset, args.normal_class, 'train', args.data_path, args.download_data, N, args.seed)


    #create train and test set
    if args.dataset =='mvtec':
        train_dataset = load_dataset(args.dataset, args.index, args.normal_class, 'train',  args.data_path, args.download_data, args.seed, N=N)
        indexes = train_dataset.indexes
    else:
        train_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'train',  args.data_path, download_data = args.download_data)
    if args.task != 'train':
        val_dataset = load_dataset(args.dataset, indexes,  args.normal_class, 'test', args.data_path, download_data=False)
    else:
        val_dataset = load_dataset(args.dataset, indexes, args.normal_class, 'validate', args.data_path, download_data=False)




    #set the seed
    torch.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed(args.weight_init_seed)
    torch.cuda.manual_seed_all(args.weight_init_seed)

    #create directories
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    if not os.path.exists('outputs/models'):
        os.makedirs('outputs/models')

    string = './outputs/models/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)
    if not os.path.exists('outputs/ED'):
        os.makedirs('outputs/ED')

    string = './outputs/ED/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/ref_vec'):
        os.makedirs('outputs/ref_vec')

    string = './outputs/ref_vec/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)

    if not os.path.exists('outputs/losses'):
        os.makedirs('outputs/losses')

    string = './outputs/losses/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/ref_vec_by_pass/'):
        os.makedirs('outputs/ref_vec_by_pass')

    string = './outputs/ref_vec_by_pass/class_' + str(args.normal_class)
    if not os.path.exists(string):
        os.makedirs(string)


    if not os.path.exists('outputs/inference_times'):
        os.makedirs('outputs/inference_times')
    if not os.path.exists('outputs/inference_times/class_' + str(args.normal_class)):
        os.makedirs('outputs/inference_times/class_'+str(args.normal_class))



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
        model = RESNET_pre( )
    elif (args.model_type == 'FASHION_VGG3'):
        if (args.pretrain ==1):
            model = FASHION_VGG3_pre(args.vector_size, args.biases)
        else:
            model = FASHION_VGG3(args.vector_size, args.biases)


    if (args.model_type == 'RESNET'):
        model.apply(deactivate_batchnorm)


    model.to(args.device)

    model_name = args.model_name + '_normal_class_' + str(args.normal_class) + '_seed_' + str(args.seed)

    #initialise the anchor
    ind = list(range(0, len(indexes)))
    #select datapoint from the reference set to use as anchor
    np.random.seed(args.epochs)
    rand_freeze = np.random.randint(len(indexes) )
    base_ind = ind[rand_freeze]
    anchor = init_feat_vec(model,base_ind , train_dataset, args.device)


    criterion = ContrastiveLoss(args.alpha, anchor, args.device, args.v)
    auc, epoch, auc_min, training_time, f1,acc, train_losses= train(model,args.lr, args.weight_decay, train_dataset, val_dataset, args.epochs, criterion, args.alpha, model_name, indexes, args.data_path, args.normal_class, args.dataset, args.smart_samp,args.k, args.eval_epoch, args.model_type, args.batch_size, num_ref_eval, num_ref_dist, args.device)
