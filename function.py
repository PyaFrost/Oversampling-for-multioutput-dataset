import time
import random
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

def oversampling_multioutput_full(df_feat,df_lab,k=5,n_iter_max=100,norm=2,verbose=False):
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
    elif (type(norm)!=float and type(norm)!=int):
        norm=2
        print("norm should be a number >1 or 'inf'\ndefault standard : 2")
    elif norm<1 :
        norm=2
        print('norm should be >1\ndefault standard : 2')
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
    pbar=tqdm(total=int(n_iter_max))
    if verbose :
        pourcent=100/q
        print('Progression :',pourcent,
              '% before the first iteration\nClass distribution  :\n    -classes reaching the threshold :',
              vec_reach_treshold,"\n    -classes not reaching the threshold :",vec_not_reach)
    ind_to_keep=[i for i in range(n) if np.where(df_lab.iloc[i].values==max(df_lab.iloc[i].values))[0][0] in vec_not_reach]
    n_iter=0
    ind_new=n

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
            if verbose :
                pourcent+=100/q
                print('Progression :',pourcent,'% after',n_iter,
                      'iterations\nClass distribution :\n    -classes reaching the threshold :',
                      vec_reach_treshold,"\n    -classes not reaching the threshold :",vec_not_reach)
        pbar.update(1)
    pbar.update(int(n_iter_max)-pbar.n)
    pbar.close()
    if n_iter==n_iter_max:
        print('Maximum number of iterations reached :',int(n_iter_max))
    elif vec_not_reach!=[]:
        print("Classes {} haven't reached the treshold".format(vec_not_reach))
    else :
        print('All classes have reached the treshold')
    print('New points :',n_iter)
    return(df_feat,df_lab)
