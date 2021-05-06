import requests

url = 'http://localhost:5000/predict_score_api'


request.get( {timeout: 1500})

r = requests.post(url,json={'Gender':1, 'Age':0, 'Marital_Status':1, 'CurrentAcc_None':0,
       'CurrentAcc_LT200':0, 'Savings_LT500':1, 'CreditHistory_none/paid':1,
       'Debtors_co-applicant':0, 'Job_unskilled-resident':0, 'NumMonths':36,
       'Telephone':0, 'Purpose_education':0, 'Purpose_furniture/equip':0,
       'CreditAmount':0.308, 'Foreignworker':1, 'Debtors_guarantor':0
	
})

print(r.json())
