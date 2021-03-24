# -*- coding: utf-8 -*-

import json
import pickle

from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

features = [
     'CreditScore',
     'Age',
     'Tenure',
     'Balance',
     'NumOfProducts',
     'HasCrCard',
     'IsActiveMember',
     'EstimatedSalary',
     'France',
     'Germany',
     'Spain',
     'Female',
     'Male'
]

# Load saved ML models
tf_model = load_model('neural')

with open('NickRoyModelv2.pkl', 'rb') as f:
    NickRoyModel = pickle.load(f)

# Load scaler info    
with open('scaler_means.json') as fin:
    scaler_means = json.load(fin)
    
with open('scaler_sigmas.json') as fin:
    scaler_sigmas = json.load(fin)


# Define function to scale json data passed to endpoint
def scale_data(data_json):
    
    for key in data_json:
        if scaler_means[key] != 0:
            data_json[key] = (data_json[key] - scaler_means[key])/scaler_means[key]
    
    return data_json


# Convert scaled json data to a numpy array
def convert_to_array(data_dict):
    
    myarray = [data_dict[key] for key in features]
    myarray = np.array(myarray)
    
    return myarray.reshape(-1, len(features))


# make app
app = Flask(__name__)
app.config["DEBUG"] = True

# define endpoints
@app.route('/', methods=['GET'])
def home():       
            
    return 'App is Healthy'


@app.route('/NickRoy', methods=['POST'])
def NickRoyModelFunction():       
        
    content = scale_data(request.json)
    data_array = convert_to_array(content)
    prediction = int((NickRoyModel.predict(data_array) > 0.5).astype(int))
    
    return jsonify(prediction)


@app.route('/neural', methods=['POST'])
def neural():       
        
    content = scale_data(request.json)
    data_array = convert_to_array(content)
    prediction = int((tf_model.predict(data_array) > 0.5).astype(int))
    
    return jsonify(prediction)

if __name__ == '__main__':
    app.run()