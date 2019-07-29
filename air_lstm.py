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
from keras.models import save_model
from keras.utils.vis_utils import plot_model

def missing_value(data):
    average=[]
    for i in range(len(data.columns)):
        sum1=0
        count=0
        for j in range(len(data)):
            if data.iloc[j][i]!='None':
                sum1+=float(data.iloc[j][i])
                count+=1
        avg=sum1/count
        average.append(avg)
    for i in range(len(data.columns)):
        for j in range(len(data)):
            if data.iloc[j][i]=='None':
                data.iloc[j][i]=average[i]
    return data

def create_dataset(data, look_back, split_frac):
    
    scaler=MinMaxScaler(feature_range=(0,1))
    data=scaler.fit_transform(data)
    
    dataX,dataY=[],[]
    for i in range(data.shape[0]-look_back):
        x=data[i:i+look_back,]
        dataX.append(x)
        y=data[i+look_back,]
        dataY.append(y)
    dataX=np.array(dataX)
    dataY=np.array(dataY)
    
    train_size = int(split_frac*data.shape[0])
    X_train = dataX[:train_size]
    y_train = dataY[:train_size]
    X_test = dataX[train_size:]
    y_test = dataY[train_size:]
    
    return X_train, y_train, X_test, y_test,scaler

def inverse_transforms(train_predict, y_train, test_predict, y_test,scaler):
    
    train_predict = scaler.inverse_transform(train_predict).reshape(y_train.shape[0],-1)
    y_train = scaler.inverse_transform(y_train)

    test_predict = scaler.inverse_transform(test_predict).reshape(y_test.shape[0],-1)
    y_test = scaler.inverse_transform(y_test)
        
    return train_predict, y_train, test_predict, y_test

def model_lstm(Input_Shape,nodes):
    
    model=Sequential()
    model.add(LSTM(nodes,input_shape=Input_Shape[1:]))
    model.add(Dropout(0.3))
    model.add(Dense(Input_Shape[2],activation='relu'))
    model.compile(optimizer='adam',loss='mean_squared_error')
    
    return model

def train_model(X_train, y_train, X_test, y_test,lstm_params):
    np.random.seed(1)

    Input_Shape=X_train.shape    
    model = model_lstm(Input_Shape,lstm_params[0])
    history=model.fit(X_train, y_train, epochs=lstm_params[2], batch_size=lstm_params[1])
    
    return model,history

def eval_model(model,X_train, y_train, X_test, y_test,scaler):
    
    # making predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    
    # inverse transforming results
    train_predict, y_train, test_predict, y_test = inverse_transforms(train_predict, y_train, test_predict, y_test, scaler)
    
    return train_predict, y_train, test_predict, y_test

def predict_test(X_test,model,scaler ):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions).reshape(X_test.shape[0],-1)
    return predictions
    
def save_Model(model,name):
    save_model(model,name+'.h5')