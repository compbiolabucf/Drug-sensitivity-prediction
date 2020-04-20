# Drug-sensitivity-prediction
User need to save input RNA-seq and target drug response files in the same folder. Name of input gene expression file is **RNASeq_log.csv** and drug response file is **Drug_AUC.csv** in all the codes. 

## **E_net.py** 

The code for Elastic net regressor that takes gene expression as input, predict drug response in terms of AUC and finally gives the correlation coefficient between the predicted AUC and actual AUC as output. User needs to specify the path to the folder where the input and target files are saved at the beginning of the code to the variable **_path_**. The variable **_FS_** sets the feature selection scheme where 0 indicates correlation based feature selection and 1 indicates network based feature selection. This code selects top 100 features in either cases which can be changed by modifying the variable **_cutoff_**. It uses the python package 

## **PLSR.py** 

The code for Partial least squares regressor that takes gene expression as input, predict drug response in terms of AUC and finally gives the correlation coefficient between the predicted AUC and actual AUC as output. User needs to specify the path to the folder where the input and target files are saved at the beginning of the code to the variable **_path_**. The variable **_FS_** sets the feature selection scheme where 0 indicates correlation based feature selection and 1 indicates network based feature selection. This code selects top 100 features in either cases which can be changed by modifying the variable **_cutoff_**.

## **RF.py** 

The code for Random forest regressor that takes gene expression as input, predict drug response in terms of AUC and finally gives the correlation coefficient between the predicted AUC and actual AUC as output. User needs to specify the path to the folder where the input and target files are saved at the beginning of the code to the variable **_path_**. The variable **_FS_** sets the feature selection scheme where 0 indicates correlation based feature selection and 1 indicates network based feature selection. This code selects top 100 features in either cases which can be changed by modifying the variable **_cutoff_**.

## **SVR.py** 

The code for Support vector machine that takes gene expression as input, predict drug response in terms of AUC and finally gives the correlation coefficient between the predicted AUC and actual AUC as output. User needs to specify the path to the folder where the input and target files are saved at the beginning of the code to the variable **_path_**. The variable **_FS_** sets the feature selection scheme where 0 indicates correlation based feature selection and 1 indicates network based feature selection. This code selects top 100 features in either cases which can be changed by modifying the variable **_cutoff_**.

Please send any email request to Dr. Taehyun Hwang (hwangt@ccf.org) to access the RNA-seq and drug response datasets.
