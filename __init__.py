from flask import Flask,jsonify
import air_lstm
import anomaly_gaussian 
import airaq
from keras.models import load_model
import pandas as pd
import tensorflow as tf
import pickle
import numpy as np

model=load_model('lstm_air.h5')
air=pd.read_csv("data\\Air.csv")
air=air.drop(['From Date','To Date'],axis=1)
air=air_lstm.missing_value(air)
X_train,Y_train,X_test,Y_test,scaler=air_lstm.create_dataset(air,24,0.9)

model_noise=load_model('lstm_noise.h5')
noise=pd.read_csv("data\\Noise.csv")
noise=noise.drop(['Timestamp','LAF'],axis=1)
noise=noise.dropna()
X_train1,Y_train1,X_test1,Y_test1,scaler1=air_lstm.create_dataset(noise,24,0.9)

model_water=load_model('lstm_water.h5')
water=pd.read_csv("data\\Water.csv")
water=water.drop(['Sampling_Date'],axis=1)
for i in water.columns:
    water[i]=water[i].fillna(value=water[i].mean())
water=anomaly_gaussian.remove_spikes(water)
X_train2,Y_train2,X_test2,Y_test2,scaler2=air_lstm.create_dataset(water,24,0.9)

with open('health_water_BOD','rb') as f:
    health_BOD=pickle.load(f)
with open('health_water_PH','rb') as f:
    health_PH=pickle.load(f)
with open('health_air','rb') as f:
    health_air=pickle.load(f)
with open('health_noise','rb') as f:
    health_noise=pickle.load(f)


global graph
graph = tf.get_default_graph()

app = Flask(__name__)

@app.route('/')
def hello():
    return jsonify({"about":"Hello World!"})

@app.route('/air',methods = ['GET'])
def air_data():
    data = {"success": False}
    test=airaq.call_aq() 
    parameters=['pm10', 'pm25', 'no2', 'co','o3', 'so2']
    with graph.as_default():
        test=scaler.transform(test).reshape((1,24,6))
        pred=air_lstm.predict_test(test,model,scaler)[0]
        data["prediction"] = dict(zip(parameters,pred.tolist()))
        data["success"] = True
    return jsonify(data)

@app.route('/noise',methods = ['GET'])
def noise_data():
    data = {"success": False}
    parameters=['C-Weighted Noise Level']
    with graph.as_default():
        test=X_test1[-1,:,:].reshape(1,24,1)
        pred=air_lstm.predict_test(test,model_noise,scaler1)[0]
        data["prediction"] = dict(zip(parameters,pred.tolist())) 
        data["success"] = True
    return jsonify(data)

@app.route('/water',methods = ['GET'])
def water_data():
    data = {"success": False}
    parameters=['pH','Conductivity','BOD','NitrateN','Fecal Coliform','Total Coliform']
    with graph.as_default():
        test=X_test2[-1,:,:].reshape(1,24,6)
        pred=air_lstm.predict_test(test,model_water,scaler2)[0]
        data["prediction"] = dict(zip(parameters,pred.tolist()))
        data["success"] = True
    return jsonify(data)
    
@app.route('/health',methods = ['GET'])
def health_data():
    data = {"success": False}
    parameters=['pH','BOD','Air','Noise']
    pred=[]
    with graph.as_default():
        pred_water=air_lstm.predict_test(X_test2[-1,:,:].reshape(1,24,6),model_water,scaler2)[0]
        pred.append(health_PH.predict(pred_water[0])[0])
        pred.append(health_BOD.predict(pred_water[2])[0])
        pred_air=air_lstm.predict_test(scaler.transform(airaq.call_aq()).reshape((1,24,6)),model,scaler)
        pred.append(health_air.predict(np.flip(pred_air[:,:2],axis=1))[0])
        pred_noise=air_lstm.predict_test(X_test1[-1,:,:].reshape(1,24,1),model_noise,scaler1)
        pred.append(health_noise.predict(pred_noise)[0])
        data["prediction"] = dict(zip(parameters,pred)) 
        data["success"] = True
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True,port=8000)
    
    