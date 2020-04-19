import torch 
import pandas as pd
import numpy as np
import sys
from torch.utils.data.dataset import Dataset
import copy
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import random
from math import floor
from torch import nn
from torch.nn import init
from tqdm import tqdm

path=

class Drug_Dataset(Dataset):
    def __init__(self, input_file, target_file, transform=None, split=None):

        self.transform = transform  # data transform
        self.split = split  # dataset split for train/val/test

        self.input_data = input_file.astype(
            np.float32, copy=False)
        self.target_data = target_file.astype(
            np.float32, copy=False)
    
    def __reorder__(self, non_test_idx=None):

        # original matrix for correlation coefficients
        if non_test_idx is None:
            mat = np.append(self.input_data, self.target_data.T, axis=0)
        else:
            mat = np.append(self.input_data[:, non_test_idx],
                                  self.target_data[non_test_idx].T, axis=0)
    
        C_mat1 = np.corrcoef(mat)
        C_shape = C_mat1.shape[0]
        corr_list = C_mat1[C_shape-1, :C_shape-1] 
        corr_index = corr_list.argsort()[::-1]
        self.input_data = self.input_data[list(corr_index), :]
        co_exp_mat=np.corrcoef(self.input_data)
        
        return co_exp_mat  

        
    def __getitem__(self,idx):

        sample = {'input_data': self.input_data[:, idx],
                  'target_data': self.target_data[idx, :]}

        if self.transform:
            sample = self.transform(sample)   

        return sample['input_data'], sample['target_data']
    
    def __len__(self):
        return len(self.target_data)
    
  
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        return {'input_data': torch.from_numpy(sample['input_data']),
                'target_data': torch.from_numpy(sample['target_data'])}
        

def data_split(dataset, split_ratio=[0.8, 0.1, 0.1], shuffle=False, manual_seed=None):

    length = dataset.__len__()
    indices = list(range(0, length))

    assert (sum(split_ratio) == 1), "Partial dataset is not used"

    if manual_seed is None:
        manual_seed = random.randint(1, 10000)
        
    if shuffle == True:
        random.seed(manual_seed)
        random.shuffle(indices)    

    breakpoint_train = floor(split_ratio[0] * length)
    breakpoint_val = floor(split_ratio[1] * length)

    idx_train = indices[:breakpoint_train]
    idx_val = indices[breakpoint_train:breakpoint_train+breakpoint_val]
    idx_test = indices[breakpoint_train+breakpoint_val:]

    return idx_train, idx_val, idx_test

      
dataset_transform = transforms.Compose([ToTensor()])

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(model, criterion, optimizer, train_loader, epoch,co_exp, learning_rate, reduction='avg',
          rank=50, print_log=True):

   
    losses = AverageMeter()

    model.train()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data = torch.autograd.Variable(data)
        target = torch.autograd.Variable(target)
        optimizer.zero_grad()
        co_exp=np.concatenate((np.concatenate((co_exp[0:rank//2,0:rank//2], co_exp[0:rank//2,-rank//2:]), axis=1), 
                                np.concatenate((co_exp[-rank//2:,0:rank//2], co_exp[-rank//2:,-rank//2:]), axis=1)), axis=0)

        C=np.sqrt(np.outer(np.sum(np.absolute(co_exp),0),np.sum(np.absolute(co_exp),1)))
        co_exp=np.divide(co_exp,C.transpose())
        
        output = model(torch.cat((data[:, 0:rank//2],data[:,-rank//2:]),axis=1),co_exp)
        loss = criterion(output, target)

        losses.update(loss.data.item(), data.size(0))

        loss.backward()
        optimizer.step()

        if print_log:
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))

    if reduction is 'sum':
        return losses.sum
    elif reduction is 'avg':
        return losses.avg



def Pearson_loss(pred, target):
    v_pred = pred - torch.mean(pred)
    v_target = target - torch.mean(target)
    corr_coe = torch.sum(v_pred * v_target) / (torch.sqrt(torch.sum(v_pred**2))
                                               * torch.sqrt(torch.sum(v_target**2)))
    return corr_coe


def corr_coef_eval(model, val_loader,co_exp, trial_idx=0, rank=50, print_log=True):

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(val_loader):
            data = torch.autograd.Variable(data)
            target = torch.autograd.Variable(target)
            co_exp=np.concatenate((np.concatenate((co_exp[0:rank//2,0:rank//2], co_exp[0:rank//2,-rank//2:]), axis=1), 
                                np.concatenate((co_exp[-rank//2:,0:rank//2], co_exp[-rank//2:,-rank//2:]), axis=1)), axis=0)

            C=np.sqrt(np.outer(np.sum(np.absolute(co_exp),0),np.sum(np.absolute(co_exp),1)))
            co_exp=np.divide(co_exp,C.transpose())
        
            output = model(torch.cat((data[:, 0:rank//2],data[:,-rank//2:]),axis=1),co_exp)
            output_var = output.std()
            corr_coef = Pearson_loss(output, target)

        if print_log:
            print('eval trial:{} \t Correlation Coefficient: {:.4f} \t Output Std {:.3f}'.format(trial_idx,
                                                                                                 corr_coef,
                                                                                                 output_var))

    return corr_coef,output,target


class Net(nn.Module):

    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
        self.hidden1 = nn.Linear(n_input, 400, bias=False)
        self.bn1 = nn.BatchNorm1d(400)
        self.sigmoid1 = nn.ReLU()

        self.hidden2 = nn.Linear(400, 100, bias=False)
        self.bn2 = nn.BatchNorm1d(100)
        self.sigmoid2 = nn.ReLU()

        self.predict = nn.Linear(100, n_output, bias=True)
        self.softmax = nn.Softmax(dim=0)
      
    def forward(self, x,co_exp):

        co_exp=torch.from_numpy(co_exp)
        out=torch.matmul(co_exp.float(),x.t())
        out = self.hidden1(out.t())
        out = self.bn1(out)
        out = self.sigmoid1(out)

        out = self.hidden2(out)
        out = self.bn2(out)
        out = self.sigmoid2(out)

        out = self.predict(out)
        out = self.softmax(out)

        return out


#### input preparation ####

input_file=path+'/RNASeq_log.csv'
output_file=path+'/Drug_AUC.csv'

data = pd.read_csv(input_file,delimiter=',',index_col=0)
celllines=np.array(data.columns)
data=np.array(data).transpose().astype(float)

c=[]
for i in range(np.size(data,1)):
    a=np.isnan(data[:,i])
    if np.sum(a)>.1*np.size(a):
        c.append(i)
    else:
        idxx=np.where(a==1)
        a = np.ma.array(data[:,i], mask=False)
        a.mask[idxx] = True
        data[idxx,i]=np.mean(a)
data=np.delete(data,c,1)  

variance_list = []
mean_list = []
for i in range(np.size(data,1)):
    values = data[:,i]
    variance_list.append(np.var(values))
    mean_list.append(np.mean(values))

mean_var = np.mean(variance_list)
mean_mean = np.mean(mean_list)

expr_data = []
for i in range(np.size(data,1)):
    if variance_list[i]>=mean_var*1.5 and mean_list[i]>=mean_mean*1.5:
        expr_data.append(data[:,i])

expr_data=np.array(expr_data).transpose()

drug_data = pd.read_csv(output_file,delimiter=',',index_col=0)
drug_name=np.array(drug_data.columns)
drug_celllines=np.array(drug_data.index)
drug_data=np.array(drug_data)

c=[]
for i in range(np.size(drug_data,1)):
    unique, counts = np.unique(drug_data[:,i], return_counts=True) 
    if np.max(counts)>.8*np.size(drug_data,0):
        c.append(i)
drug_data=np.delete(drug_data,c,1)
drug_name=np.delete(drug_name,c)

xy, x_ind, y_ind = np.intersect1d(celllines,drug_celllines,return_indices=True)

expr_data=expr_data[x_ind,:].transpose()
drug_data=drug_data[y_ind,:]

### model ###

N_input = 100 
N_target = 1

LR = 0.01  # learning rate
N_EPOCHS = 100
print_log = False
N_trial = 50

output=[]
for drug_idx in range(np.size(drug_name)):
    drugy = drug_data[:,drug_idx]
    a=[i for i, x in enumerate(drugy) if not str(x).replace('.','').isdigit()]
    y=np.delete(drugy,a)
    y=np.expand_dims(y, axis=1)
    X=np.delete(expr_data,a,axis=1)

    dataset = Drug_Dataset(X,y,transform=dataset_transform)

    roc_drug=[]
    correlation=[]
    for idx_trial in tqdm(range(N_trial)):

        model = Net(N_input, N_target)

        Criterion = nn.MSELoss()
        Optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=0.9, nesterov=True,
                                    weight_decay=0.001)
 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(Optimizer, 0.99, last_epoch=-1)
        train_idx, val_idx, test_idx = data_split(dataset, split_ratio=[0.46, 0.20, 0.34],
                                                  shuffle=True, manual_seed=None)

        co_exp=dataset.__reorder__(train_idx+val_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                   batch_size=train_idx.__len__(),
                                                   sampler=train_sampler,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   pin_memory=True)


        val_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                 batch_size=val_idx.__len__(),
                                                 sampler=val_sampler,
                                                 shuffle=False,
                                                 num_workers=0,
                                                 pin_memory=True)

        test_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=test_idx.__len__(),
                                                  sampler=test_sampler,
                                                  shuffle=False,
                                                  num_workers=0,
                                                  pin_memory=True)
    
        best_corr_coef = None

        for epoch in range(N_EPOCHS):

            train_loss = train(model, Criterion, Optimizer, train_loader, epoch,co_exp,
                           learning_rate=LR,rank=N_input, print_log=print_log)

            val_corr_coef,out,tar = corr_coef_eval(model, val_loader,co_exp, rank=N_input, trial_idx=idx_trial,
                                           print_log=print_log)

            if best_corr_coef is None:
                is_best = True
                best_corr_coef = val_corr_coef
            else:
                is_best = val_corr_coef > best_corr_coef
                best_corr_coef = max(val_corr_coef, best_corr_coef)

            if is_best:  # make a copy of the best model
                model_best = copy.deepcopy(model)

            scheduler.step()  # update the learning rate with scheduler
    
        test_corr_coef,out,tar = corr_coef_eval(model_best, test_loader,co_exp, rank=N_input, trial_idx=idx_trial,
                                        print_log=print_log)
    
        correlation.append(test_corr_coef)

    print(drug_idx,' ',drug_name[drug_idx],':',np.mean(correlation)) 
    output.append(correlation)
    if ((drug_idx%10)==0):
        print(np.mean(output))
    
print(np.shape(output)) 

np.savetxt('corr_nn_2_coexp_100_6631.csv',np.transpose(output),fmt='%s',delimiter=',')  

