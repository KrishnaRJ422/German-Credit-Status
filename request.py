import requests

url = 'http://localhost:5000/predict_score_api'


request.get( {timeout: 1500})

r = requests.post(url,json={'CurrentAcc_None':0, 'NumMonths':36, 'CreditHistory_Delay':1,
       'CreditHistory_none/paid':0, 'Collateral_savings/life_insurance':0,
       'CurrentAcc_GE200':1, 'Purpose_repairs':0, 'Purpose_radio/tv':0, 'Gender':1,
       'Age':1
		
})

print(r.json())
