import openaq
import json
import pandas as pd
import numpy as np

def call_aq():
    api=openaq.OpenAQ()
    parameters=['pm10', 'pm25', 'no2', 'co','o3', 'so2']
    data_new=pd.DataFrame()
    for i in parameters:
        status,resp=api.measurements(city='Delhi',location='Anand Vihar, Delhi - DPCC',parameter=[i],limit=24)
        json_dict=json.dumps(resp['results'])
        data=pd.read_json(json_dict)
        data=data['value']
        data=data.to_frame()
        data=data.rename(columns={'value': i})
        data_new=pd.concat([data_new,data],axis=1)
    
    data_new.iloc[:,3]/=1000
    data_new.iloc[:,2]/=10
    test=np.array(data_new)
    return test

#test=call_aq()
#print(test.shape)
#test=scaler.transform(test).reshape(1,24,6)
#print(air_lstm.predict_test(test,model,scaler))
