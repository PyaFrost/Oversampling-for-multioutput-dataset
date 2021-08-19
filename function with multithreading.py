import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing.pool import Pool
from sklearn.model_selection import train_test_split

def oversampling_multioutput_low(df_feat,df_lab,n_para=None,k=5,n_iter_max=100,norm=2):
    n=df_lab.shape[0]
    q=df_lab.shape[1]
    columns_df_lab=list(df_lab.columns)
    columns_df_feat=list(df_feat.columns)
    v_sum=sum(df_lab.values)
    treshold=max(v_sum)
    ind_treshold=np.where(v_sum==treshold).index(0)
    vec_reach_treshold=[ind_treshold]
    vec_not_reach=[i for i in range(q)]
    del vec_not_reach[ind_treshold]
    v_sum_not_reach=list(np.copy(v_sum))
    del v_sum_not_reach[ind_treshold]
    v_sum_not_reach=np.array(v_sum_not_reach)
    ind_to_keep=[i for i in range(n) if np.where(df_lab.iloc[i].values==max(df_lab.iloc[i].values))[0][0] in vec_not_reach]
    n_iter=0
    ind_new=n_para
    while n_iter<n_iter_max and vec_not_reach!=[] and ind_to_keep!=[]:
        n_iter+=1
        r=int(random.choice(ind_to_keep))
        base_point_lab=df_lab.iloc[r].values
        base_point_feat=df_feat.iloc[r].values
        k_n=np.empty((k,2))
        ind_to_use=[i for i in ind_to_keep if i!=r]
        j=0
        for i in ind_to_use[0:k]:
            k_n[j]=[ind_to_use[j],np.linalg.norm(base_point_feat-df_feat.iloc[i].values,norm)]
            j+=1
        m=max(k_n[:,1])
        for i in ind_to_use[k:]:
            d=np.linalg.norm(base_point_feat-df_feat.iloc[i].values,norm)
            if d<m:
                k_n[np.where(k_n==m)[0][0],:]=[i,d]
                m=max(k_n[:,1])
        r_2=random.randint(0,k-1)
        random_neighboor_lab=df_lab.iloc[int(k_n[r_2,0])].values
        random_neighboor_feat=df_feat.iloc[int(k_n[r_2,0])].values
        w=random.random()
        newpoint_feat=w**2*random_neighboor_feat+(1-w**2)*base_point_feat
        newpoint_lab=w**2*random_neighboor_lab+(1-w**2)*base_point_lab
        #Ajout du point aux data-frames
        new_feat_row=pd.DataFrame([tuple(newpoint_feat)],columns=columns_df_feat,index=[ind_new])
        df_feat=df_feat.append(new_feat_row,ignore_index=False)
        new_lab_row=pd.DataFrame([tuple(newpoint_lab)],columns=columns_df_lab,index=[ind_new])
        df_lab=df_lab.append(new_lab_row,ignore_index=False)
        ind_new+=1
        v_sum_not_reach=np.add(v_sum_not_reach,newpoint_lab[vec_not_reach])
        v_sum=np.add(v_sum,newpoint_lab)
        if any(item>treshold for item in v_sum_not_reach):
            ind_reach=np.where(v_sum_not_reach>treshold)[0]
            vec_reach_treshold=sorted(np.append(vec_reach_treshold,np.array(vec_not_reach)[ind_reach]))
            vec_not_reach = [i for i in vec_not_reach if i not in np.array(vec_not_reach)[ind_reach]]
            v_sum_not_reach=v_sum[vec_not_reach]
            ind_to_keep=[i for i in ind_to_keep if np.where(df_lab.iloc[i].values==max(df_lab.iloc[i].values))[0][0] in vec_not_reach]
    return(df_feat,df_lab)

#######################################################################################################################################

def oversampling_multioutput_para(df_feat,df_lab,k=5,n_iter_max=100,norm=2,n_jobs=8):
    print('Operating oversampling multithreading ...', end=' ')
    start_time=time.time()
    if type(df_feat)!=pd.core.frame.DataFrame:
        raise Exception('df_feat must be a DataFrame')
    elif type(df_lab)==pd.core.frame.DataFrame:
        n=df_lab.shape[0]
        q=df_lab.shape[1]
    elif type(df_lab)==list:
        n=len(df_lab)
        q=len(df_lab[0])
        index_df_lab=[str(i) for i in range(n)]
        columns_df_lab=['label_'+str(i) for i in range(q)]
        df_lab = pd.DataFrame(df_lab, index = index_df_lab, columns = columns_df_lab)
    else :
        raise Exception('df_lab must be a numpy.array or a DataFrame')
    if df_feat.shape[0]!=n:
        raise Exception("df_feat and df_lab don't have same number of observations")
    if norm=='inf':
        norm=np.inf
    elif (type(norm)!=float and type(norm)!=int and norm<1):
        norm=2
        print("\nnorm should be a number >1 or 'inf'\ndefault standard : 2")
    if n_jobs>1:
        len_set=n//n_jobs
        rest=n%n_jobs
        ind_usbl=[int(item) for item in df_lab.index.to_list()]
        ind_0=random.sample(ind_usbl,len_set+rest)
        list_df_feat=[df_feat.iloc[ind_0].copy()]
        list_df_lab=[df_lab.iloc[ind_0].copy()]
        ind_usbl=[i for i in ind_usbl if i not in ind_0]
        for i in range(n_jobs-1):
            ind_tmp=random.sample(ind_usbl,len_set)
            list_df_feat.append(df_feat.iloc[ind_tmp].copy())
            list_df_lab.append(df_lab.iloc[ind_tmp].copy())
            ind_usbl=[i for i in ind_usbl if i not in ind_tmp]
        p = Pool()
        args=[(list_df_feat[i],list_df_lab[i],n,k,n_iter_max,norm) for i in range(len(list_df_feat))]
        with p:
            all_df = p.starmap(oversampling_multioutput_low, args)
        p.close()
        p.join()
        new_df_feat=all_df[0][0]
        new_df_lab=all_df[0][1]
        for i in range(n_jobs-1):
            new_df_feat=new_df_feat.append(all_df[i+1][0])
            new_df_lab=new_df_lab.append(all_df[i+1][1])
    else : 
        new_df_feat,new_df_lab,a=oversampling_multioutput_low(df_feat,df_lab,n_para=None,k=k,n_iter_max=n_iter_max)
    print('Done in {}s'.format(round(time.time()-start_time),3))
    print('New points :',len(new_df_feat)-n)
    return(new_df_feat,new_df_lab)
