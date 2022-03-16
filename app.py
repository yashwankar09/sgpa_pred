# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:15:27 2022

@author: yashw
"""

import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('sgpa_pred.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prediction = np.round(prediction[0],2)
    return render_template('index.html', prediction_text='Your Predicted SGPA is {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)
