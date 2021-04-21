import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import sys
import logging

root = logging.getLogger()
handler = logging.StreamHandler(sys.stdout)
root.addHandler(handler)

app = Flask(__name__, template_folder='templates')
app.config['EXPLAIN_TEMPLATE_LOADING'] = True
model = pickle.load(open('dir_age_debiased.pkl', 'rb'))

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def home():
    return render_template('index - Copy.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
	For rendering results on HTML GUI
	'''
    print('Enter input values')
    int_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([int_features])
    final_features.columns=['Gender','Age','Marital_Status','NumMonths','Savings_<500','Savings_none','Dependents','Property_rent',
                           'Job_management/self-emp/officer/highly qualif emp','Debtors_guarantor','Purpose_CarNew',
                           'Purpose_furniture/equip','CreditHistory_none/paid','Purpose_CarUsed','CreditAmount',
                           'Collateral_real estate','Debtors_none','Job_unemp/unskilled-non resident','Purpose_others',             
                            'CreditHistory_other','PayBackPercent','Collateral_unknown/none','Purpose_education']
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    
    if output==1:
        return render_template('index - Copy.html', prediction_text='Good credit score')
    elif output==0:
        return render_template('index - Copy.html', prediction_text='Bad credit score')
    else:
        return render_template('index - Copy.html', prediction_text='Please enter input values') 
     


if __name__ == "__main__":
    app.run(debug=True)
    
