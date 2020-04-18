import pandas as pd
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import sys
import numpy.ma as ma

iteration=50
cutoff=100
path=
FS = 1                     ####### 0 for CC based, 1 for GB
alpha = np.array([.95])


def normalnp(data, alpha, F_0):
    N = data
    [m, n] = N.shape
    W = abs(np.corrcoef(N,rowvar=False))
    Sum = W.sum(axis=0)
    SumT = np.transpose(np.array(Sum))
    #S = pandas.DataFrame(0.00, index=np.arange(n), columns=np.arange(n))
    sqr = np.sqrt(np.outer(np.array(SumT), np.array(Sum)))
    S = np.divide(W, sqr)
    S = np.nan_to_num(S)
    Y = F_0
    F = F_0
    for i in range(0, 10000):
        F_old = F   
        F = np.dot(alpha[0]*F_old,S) + (1-alpha[0])*Y
        if(np.amax(abs(np.subtract(F,F_old))) < 1e-8):
            break
    if (i==9999):
        print("No converge")

    return np.transpose(F)


#######    pre processings   ########
input_file=path+'/RNASeq_log.csv'
target_file=path+'/Drug_AUC.csv'

data = pd.read_csv(input_file,delimiter=',',index_col=0)
celllines=np.array(data.columns)
data=np.array(data).transpose().astype(float)

##### removing genes that has more than 10% NaNs otherwise replace NaNs with mean expression

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

##### removing genes with low expression and variance 
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

###### processing of target file #########
drug_data = pd.read_csv(target_file,delimiter=',',index_col=0)
drug_name=np.array(drug_data.columns)
drug_celllines=np.array(drug_data.index)
drug_data=np.array(drug_data)

#### removing drugs that has same response in more than 80% samples 
c=[]
for i in range(np.size(drug_data,1)):
    unique, counts = np.unique(drug_data[:,i], return_counts=True) 
    if np.max(counts)>.8*np.size(drug_data,0):
        c.append(i)
drug_data=np.delete(drug_data,c,1)
drug_name=np.delete(drug_name,c)

xy, x_ind, y_ind = np.intersect1d(celllines,drug_celllines,return_indices=True)

X1=expr_data[x_ind,:]
drug_data=drug_data[y_ind,:]

corr_all=[]
for j in range(np.size(drug_data,1)):
  corr_drug=[]                           ####### removing cell lines with response NaN
  y1 = drug_data[:,j]
  a=[i for i, x in enumerate(y1) if not str(x).replace('.','').isdigit()]
  y=np.delete(y1,a)
  X=np.delete(X1,a,axis=0)
  
  for i in range(iteration):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=i)  
    
    if FS==1:
      cc = np.zeros((1,np.size(X,1)))
      for j in range(np.size(X,1)):
        D= pearsonr(ma.masked_invalid(X_train[:,j]), ma.masked_invalid(y_train))[0]
        cc[0,j] = D
      cc = np.nan_to_num(cc)
      F = normalnp(X, alpha, cc)
      sort=np.argsort(F,0)
      sort=sort[::-1][:,0]
      index=np.concatenate([sort[0:cutoff//2],sort[-cutoff//2:]])
    
    elif FS==0:
      CC=[pearsonr(X_train[:, i], y_train)[0] for i in range(np.size(X_train,1))]
      sort=np.argsort(np.abs(CC))
      sort=sort[::-1]
      index=sort[0:cutoff]
    else:
      print('Invalid feature selection')

    X_train=X_train[:,index]
    X_test=X_test[:,index]    
   
    regressor = PLSRegression(n_components=1)
    regressor.fit(X1_train, y_train)
    y_pred = regressor.predict(X1_test)
    y_pred=y_pred.ravel()

    corr, _ = pearsonr(y_test,y_pred)
    corr_drug.append(corr)
  corr_all.append(corr_drug)
  print(np.nanmean(corr_drug))  
print(np.nanmean(corr_all))   

np.savetxt('1PLSR_gene_100_6631_gb.csv',corr_all,delimiter=',',fmt='%s')

