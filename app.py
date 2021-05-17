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
model = pickle.load(open('xg_hp_dir_age_debiased_upd.pkl', 'rb'))

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)


@app.route('/')
def home():
    return render_template('index - Copy.html')

@app.route('/credit_status',methods=['POST'])
def predict():
    '''
	For rendering results on HTML GUI
	'''
    print('Enter input values')
    int_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame([int_features])
    final_features.columns=['CurrentAcc_None', 'NumMonths', 'CreditHistory_Delay',
       'CreditHistory_none/paid', 'Collateral_savings/life_insurance',
       'CurrentAcc_GE200', 'Purpose_repairs', 'Purpose_radio/tv', 'Gender',
       'Age']
    prediction = model.predict(np.array(final_features))
    output = round(prediction[0], 2)
    
    if output==1:
        return render_template('index - Copy.html', prediction_text='Good credit score')
    elif output==0:
        return render_template('index - Copy.html', prediction_text='Bad credit score')
    else:
        return render_template('index - Copy.html', prediction_text='Please enter input values') 
     


if __name__ == "__main__":
	app.run(debug=False,host = '127.0.0.1', port = 8080)
	#app.run()
		
	
	
#app.run(debug=True)
    
