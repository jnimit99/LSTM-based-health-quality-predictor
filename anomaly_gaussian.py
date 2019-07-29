import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense

def Mean_Variance(data):
    data_mean_df=data.mean()
    data_mean=np.array(data_mean_df).reshape(1,-1)
    data_variance=np.mean(np.square(np.array(data)-data_mean),axis=0)
    return data_mean.ravel(),data_variance.ravel()

def probability(instance,data_mean,data_variance):
    normalized_val=np.square(instance-data_mean)/(2*data_variance)
    prob=np.exp(-normalized_val)/(np.sqrt(2*np.pi)*data_variance)
    return prob

def outlier_detection(data):
    data_mean,data_variance=Mean_Variance(data)
    features=data.shape[1]
    outlier=[[]*features for i in range(features)]
    for i in range(len(data.columns)):
        for j in range(len(data)):
            outlier[i].append(probability(data.iloc[j][i],data_mean[i],data_variance[i]))
    outlier=np.array(outlier).T
    return outlier

def get_epsilon(data):
    outlier=outlier_detection(data)
    data_mean,data_variance=Mean_Variance(pd.DataFrame(outlier))
    epsilon=data_mean
    return epsilon,outlier

def remove_spikes(temp):
    epsilon,outlier=get_epsilon(temp)
    for i in range(len(temp.columns)):
        for j in range(len(temp)):
            if(outlier[j,i]<epsilon[i]):
                temp.iloc[j][i]=np.NaN
    for i in temp.columns:
        temp[i]=temp[i].fillna(value=temp[i].mean())
        
    return temp