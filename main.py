# -*- coding: utf-8 -*-
import base64
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
import pandas as pd
import xgboost as xgb
import statsmodels.api as sm
import joblib
import json
from pathlib import Path
from src.modelamiento_otros import prediccionMallaModelo
import plotly as plotly
import plotly.graph_objects as go

original_data = pd.read_csv('./data/010_DFT1b1.csv')
mean_data = pd.read_csv('./data/010_Media_Datos.csv', index_col='variable')
data_dictionary = pd.read_csv('./data/010_DFT1b1_dictionary.csv', index_col='variable', encoding='latin1')

data_dictionary.index = map(lambda x:str.lower(x), data_dictionary.index)
api = Flask(__name__)


model_route = Path('./models/')
model_path = {
    'clarity_LM'  : '050_Lineal.sav',
    'clarity_RF'  : '050_rf_1.sav',
    'clarity_SVM' : '050_SVM1.sav',
    'clarity_XGB' : '050_XGB1.sav',
    
    'bloom_LM'    : '051_Lineal.sav',
    'bloom_RF'    : '051_rf_1.sav',
    'bloom_SVM'   : '051_SVM1.sav',
    'bloom_XGB'   : '051_XGB1.sav',

    'viscosity_LM'    : '052_Lineal.sav',
    'viscosity_RF'    : '052_rf_1.sav',
    'viscosity_SVM'   : '052_SVM1.sav',
    'viscosity_XGB'   : '052_XGB1.sav',

    'yield_XGB'       : '053_Yield_XGB2.sav',
}

def modelSelection(output, selection):
    selector = "{0}_{1}".format(str.lower(output), str.upper(selection))

    model = joblib.load(open(model_route / model_path[selector], 'rb'))

    if str.upper(selection) in ['XGB', 'RF']:
        model = model.best_estimator_

    return model




@api.route('/api/predict', methods=['POST'])
def predict():
    """
    POST method for prediction.
    The data most contain information about the model.
    """

    # Get data as JSON from POST
    # print('HOLA API')

    data = request.get_json()
    # print(data)

    # Build input data
    test  = pd.DataFrame(data['data'], index=[1])
    test1 = test.copy()

    for i in test.columns:
        mn = test.loc[:, i]/mean_data.loc[i,'value']
        test1[i] = mn

    model = modelSelection(data['output'], data['model'])
    # print(model)

    # In case of intercept
    if data['intercept'] is True:
            test1 = sm.add_constant(test1, has_constant='add')
    
    # Predict using DL model
    try:
        prediction = model.predict(test1)
    except:
        return "The prediction was unsuccessfull"
    #print(prediction)

    # Send response
    message = {
        "status": 200,
        "message": [
            {
                "task": "Prediction Transaction",
                "prediction": float(np.array(prediction)[0])
            }
        ]
    }
    print( np.array(prediction)[0] )
    response = jsonify(message)
    response.status_code = 200

    return response


@api.route('/api/surface_response', methods=['POST'])
def surface_response():
    # Get data as JSON from POST
    data = request.get_json()
    

    variables = data['variables']
    original_data1 = original_data.loc[:, variables].copy()

    for i in original_data1.columns:
        mn = original_data1.loc[:, i]/mean_data.loc[i,'value']
        original_data1[i] = mn
    

    model = modelSelection(data['output'], data['model'])
    var1 = data['var1']
    var2 = data['var2']

    M = prediccionMallaModelo(original_data1, model, variables, var1, var2, k=100, normal='other')
    # print(M)
    Z = M['Z']
    X = M['X'] * mean_data.loc[var1,'value']
    Y = M['Y'] * mean_data.loc[var2,'value']
    
    xlabel = data_dictionary.loc[str.lower(var1), 'long_name']
    ylabel = data_dictionary.loc[str.lower(var2), 'long_name']
    zlabel = data_dictionary.loc[str.lower(data['output']), 'long_name']
    
    results = {
        'X': X.tolist(), 
        'Y': Y.tolist(), 
        'Z': Z.tolist(),
        'xlabel': xlabel, 'ylabel': ylabel, 'zlabel': zlabel}

    # Send response
    message = {
        "status": 200,
        "message": [
            {
                "task": "Surface 3D plot",
                "prediction": json.dumps(results)
            }
        ]
    }
    response = jsonify(message)
    response.status_code = 200

    return response

    #return response













@api.route('/api/status', methods=['GET'])
def status():
    """
    GET method for API status verification.
    """

    message = {
        "status": 200,
        "message": [
            "This API is up and running!"
        ]
    }
    response = jsonify(message)
    response.status_code = 200

    return response


@api.errorhandler(404)
def not_found(error=None):
    """
    GET method for not found routes.
    """

    message = {
        "status": 404,
        "message": [
            "[ERROR] URL not found."
        ]
    }
    response = jsonify(message)
    response.status_code = 404

    return response


if __name__ == '__main__':
    api.run(port=5000, debug=False, host= '0.0.0.0')
