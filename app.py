from multiprocessing import context
from unicodedata import name
from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route('/')
@app.route('/<name>')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/predict')
def predict():
    loaded_model = pickle.load(open('lrmodel.sav', 'rb'))


    patient_dict = {'age': 60.0, 'anaemia': 0.0, 'creatinine_phosphokinase': 235.0, 'diabetes': 1.0, 'ejection_fraction': 38.0, 'high_blood_pressure': 0.0,
                'platelets': 329000.0, 'serum_creatinine': 3.0, 'serum_sodium': 142.0, 'sex': 0.0, 'smoking': 0.0, 'time': 30.0}

    patient_df = pd.DataFrame([patient_dict])


    patient_pred = loaded_model.predict(patient_df)
    heart_failure_prob = loaded_model.predict_proba(patient_df)[:,1]

    return render_template('results.html', 
                            context= {'prediction': patient_pred, 
                                            'probability': heart_failure_prob}
            )
