

import pandas as pd
import numpy as np
import seaborn               as sns
import matplotlib.pyplot     as plt
from sklearn.model_selection import train_test_split
#from sklearn.ensemble        import RandomForestClassifier
#from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import MinMaxScaler, StandardScaler
from sklearn.base            import TransformerMixin
from sklearn.pipeline        import Pipeline, FeatureUnion
from typing                  import List, Union, Dict
# Warnings will be used to silence various model warnings for tidier output
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
np.random.seed(0)
from sklearn.model_selection import GridSearchCV
#!pip install fairlearn
#!pip install aif360
#!pip install shap
#!pip install eli5
#!pip install BlackBoxAuditing

German_df = pd.read_csv('C:/Users/krish/Downloads/German-reduced_upd.csv')

print(German_df.shape)
print (German_df.columns)



German_df.head()




feature_list=['CurrentAcc_None', 'NumMonths', 'CreditHistory_Delay',
       'CreditHistory_none/paid', 'Collateral_savings/life_insurance',
       'CurrentAcc_GE200', 'Purpose_repairs', 'Purpose_radio/tv', 'Gender',
       'Age', 'CreditStatus']



X = German_df.iloc[:, :-1]
y = German_df['CreditStatus']
X.head()
y.head()


German_df.head()


# German_df.columns=feature_list
# German_df.head()

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric

def fair_metrics( dataset, pred, pred_is_dataset=False):
    if pred_is_dataset:
        dataset_pred = pred
    else:
        dataset_pred = dataset.copy()
        dataset_pred.labels = pred

    cols = ['Accuracy', 'F1', 'DI','SPD', 'EOD', 'AOD', 'ERD', 'CNT', 'TI']
    obj_fairness = [[1,1,1,0,0,0,0,1,0]]

    fair_metrics = pd.DataFrame(data=obj_fairness, index=['objective'], columns=cols)

    for attr in dataset_pred.protected_attribute_names:
        idx = dataset_pred.protected_attribute_names.index(attr)
        privileged_groups =  [{attr:dataset_pred.privileged_protected_attributes[idx][0]}]
        unprivileged_groups = [{attr:dataset_pred.unprivileged_protected_attributes[idx][0]}]

        classified_metric = ClassificationMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        metric_pred = BinaryLabelDatasetMetric(dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        distortion_metric = SampleDistortionMetric(dataset,
                                                     dataset_pred,
                                                     unprivileged_groups=unprivileged_groups,
                                                     privileged_groups=privileged_groups)

        acc = classified_metric.accuracy()
        f1_sc = 2 * (classified_metric.precision() * classified_metric.recall()) / (classified_metric.precision() + classified_metric.recall())

        mt = [acc, f1_sc,
                        classified_metric.disparate_impact(),
                        classified_metric.mean_difference(),
                        classified_metric.equal_opportunity_difference(),
                        classified_metric.average_odds_difference(),
                        classified_metric.error_rate_difference(),
                        metric_pred.consistency(),
                        classified_metric.theil_index()
                    ]
        w_row = []
        print('Computing fairness of the model.')
        for i in mt:
            row = pd.DataFrame([mt],
                           columns  = cols,
                           index = [attr]
                          )
        fair_metrics = fair_metrics.append(row)
    fair_metrics = fair_metrics.replace([-np.inf, np.inf], 2)
    return fair_metrics

def get_fair_metrics_and_plot( data, model, plot=False, model_aif=False):
    pred = model.predict(data).labels if model_aif else model.predict(data.features)
    fair = fair_metrics( data, pred)
    if plot:
        pass

    return fair

def get_model_performance(X_test, y_true, y_pred, probs):
    accuracy = accuracy_score(y_true, y_pred)
    matrix = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, matrix, f1

def plot_model_performance(model, X_test, y_true):
    y_pred = model.predict(X_test)
    probs = model.predict_proba(X_test)
    accuracy, matrix, f1 = get_model_performance(X_test, y_true, y_pred, probs)

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer
from aif360.metrics import ClassificationMetric
# Get DF into IBM format
from aif360 import datasets
#converting to aif dataset
aif_dataset = datasets.BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=German_df,
                                                      label_names=["CreditStatus"],
                                                     protected_attribute_names=["Age"],
                                              privileged_protected_attributes = [1])

privileged_groups = [{'Age': 1}]
unprivileged_groups = [{'Age': 0}]

data_orig_train, data_orig_test = aif_dataset.split([0.7], shuffle=True)

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()

X_train.shape
X_test.shape

data_orig_test.labels[:10].ravel()

data_orig_train.labels[:10].ravel()

metric_orig_train = BinaryLabelDatasetMetric(data_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


filename= 'C:/Users/krish/Downloads/main_pjt_final - Copy/filename_mainpjt_results_age_may6_jp.csv'


# ## <font color='crimson'>Converting data to aif compatible format</font>

# Since we are dealing with binary label dataset we are using aif360 class BiaryLabelDataset here with target label as CreditStatus and protected attributes as age,gender,marital status. Refer part 11 for more details on protected attributes and privileged classes.


# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.explainers import MetricTextExplainer
from aif360.metrics import ClassificationMetric
# Get DF into IBM format
from aif360 import datasets
#converting to aif dataset
aif_dataset = datasets.BinaryLabelDataset(favorable_label = 1, unfavorable_label = 0, df=German_df,
                                                      label_names=["CreditStatus"],
                                                     protected_attribute_names=["Age"],
                                              privileged_protected_attributes = [1])





privileged_groups = [{'Age': 1}]
unprivileged_groups = [{'Age': 0}]



data_orig_train, data_orig_test = aif_dataset.split([0.8], shuffle=True)

X_train = data_orig_train.features
y_train = data_orig_train.labels.ravel()

X_test = data_orig_test.features
y_test = data_orig_test.labels.ravel()




X_train.shape
X_test.shape




data_orig_test.labels[:10].ravel()




data_orig_train.labels[:10].ravel()





metric_orig_train = BinaryLabelDatasetMetric(data_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())


# A non zero value indicates bias.



from xgboost import XGBClassifier
estimator = XGBClassifier(seed=40)

parameters = {
    'max_depth': range (2, 10, 2),
    'n_estimators': range(60, 240, 40),
    'learning_rate': [0.1, 0.01, 0.05]
}
grid_search = GridSearchCV(
    estimator=estimator,
    param_grid=parameters,
    scoring = 'recall',
    
    cv = 5,
    verbose=0
)

model_xg=grid_search




mdl_xgb = model_xg.fit(data_orig_train.features, data_orig_train.labels.ravel())



from sklearn.metrics import confusion_matrix
conf_mat_xg = confusion_matrix(data_orig_test.labels.ravel(), model_xg.predict(data_orig_test.features))
conf_mat_xg
from sklearn.metrics import accuracy_score
print(accuracy_score(data_orig_test.labels.ravel(), model_xg.predict(data_orig_test.features)))


import pandas as pd
import csv
import os
import numpy as np
import sys
from aif360.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
plot_model_performance(mdl_xgb, X_test, y_test)






from aif360.algorithms.preprocessing import DisparateImpactRemover

DIR_xg = DisparateImpactRemover()
data_transf_train_xg_dir = DIR_xg.fit_transform(data_orig_train)

# Train and save the model
xg_transf_dir = model_xg.fit(data_transf_train_xg_dir.features,data_transf_train_xg_dir.labels.ravel())


# * If the measure is less than or greater than 0,bias exists. The negative bias denotes that the prediction is biased
# towards privileged group and positive bias denotes that prediction is biased towards unprivileged group.

import numpy as np
import pickle
pickle.dump(xg_transf_dir,open('xg_hp_dir_age_debiased_upd.pkl','wb'))
xg_dir_model_age=pickle.load(open('xg_hp_dir_age_debiased_upd.pkl','rb'))
data_orig_test.features[0]
data_orig_test.labels[0].ravel()
data_orig_test.features[0]
pred=xg_transf_dir.predict(np.array([[ "0", "36",  "0",  "1",  "0",  "1",  "0",  "1",  "1",  "0"    ]]))
pred[0]