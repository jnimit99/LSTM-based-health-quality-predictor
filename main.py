import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM,Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os
from air_lstm import * 
from anomaly_gaussian import *

if __name__ == '__main__':
    
    air=pd.read_csv("C:\\Users\\hp\\Downloads\\Air Pollution\\JuneToAugust.csv")
    air=air.drop(['From Date','To Date'],axis=1)
    air=missing_value(air)
    
    noise=pd.read_csv("C:\\Users\\hp\\Downloads\\Air Pollution\\NoiseCPCB.csv")
    noise=noise.drop(['Timestamp','LAF'],axis=1)
    noise=noise.dropna() 
    
    water=pd.read_csv("C:\\Users\\hp\\Downloads\\Air Pollution\\Water.csv")
    water=water.drop(['Sampling_Date'],axis=1)
    for i in water.columns:
        water[i]=water[i].fillna(value=water[i].mean())
    water=remove_spikes(water)
    
    look_back = 24
    split = 0.77
    nodes = 10
    epochs = 10 
    batch_size=1 
    lstm_params = [nodes, batch_size,epochs]
    
    X_train,Y_train,X_test,Y_test,scaler=create_dataset(air,look_back,split)
    model,history= train_model(X_train,Y_train,X_test,Y_test, lstm_params)
    train_predict, y_train, test_predict, y_test =eval_model(model,X_train, Y_train, X_test,Y_test,scaler)

    # calculating RMSE metrics
    error = np.sqrt(mean_squared_error(train_predict, y_train))
    print('Train RMSE: %.3f' % error)
    error = np.sqrt(mean_squared_error(test_predict, y_test))
    print('Test RMSE: %.3f' % error)
    
    save_Model(model,'lstm_air')
    
    X_train,Y_train,X_test,Y_test,scaler=create_dataset(noise,look_back,split)
    model,history= train_model(X_train,Y_train,X_test,Y_test, lstm_params)
    train_predict, y_train, test_predict, y_test =eval_model(model,X_train, Y_train, X_test,Y_test,scaler)

    # calculating RMSE metrics
    error = np.sqrt(mean_squared_error(train_predict, y_train))
    print('Train RMSE: %.3f' % error)
    error = np.sqrt(mean_squared_error(test_predict, y_test))
    print('Test RMSE: %.3f' % error)
    
    save_Model(model,'lstm_noise')
    
    X_train,Y_train,X_test,Y_test,scaler=create_dataset(water,look_back,split)
    model,history= train_model(X_train,Y_train,X_test,Y_test, lstm_params)
    train_predict, y_train, test_predict, y_test =eval_model(model,X_train, Y_train, X_test,Y_test,scaler)

    # calculating RMSE metrics
    error = np.sqrt(mean_squared_error(train_predict, y_train))
    print('Train RMSE: %.3f' % error)
    error = np.sqrt(mean_squared_error(test_predict, y_test))
    print('Test RMSE: %.3f' % error)
    
    save_Model(model,'lstm_water')
    
    
    


