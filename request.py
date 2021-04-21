import requests

url = 'http://localhost:5000/predict_score_api'


request.get( {timeout: 1500})

r = requests.post(url,json={'Gender':1,'Age':1,'Marital_Status':1,'NumMonths':24,'Savings_<500':1,'Savings_none':0,'Dependents':1,'Property_rent':0,
                           'Job_management/self-emp/officer/highly qualif emp':0,'Debtors_guarantor':0,'Purpose_CarNew':0,
                           'Purpose_furniture/equip':0,'CreditHistory_none/paid':0,'Purpose_CarUsed':0,'CreditAmount':0.04258831,
                           'Collateral_real estate':1,'Debtors_none':1,'Job_unemp/unskilled-non resident':0,'Purpose_others':0,             
                            'CreditHistory_other':0,'PayBackPercent':4,'Collateral_unknown/none':0,'Purpose_education':0
	
})

print(r.json())
