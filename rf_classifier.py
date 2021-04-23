# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:19:30 2021

@author: krish
"""

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
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
np.random.seed(0)

## <font color='crimson'> Importing source dataset </font>

German_df = pd.read_csv('German-reduced.csv')

print(German_df.shape)
print (German_df.columns)

German_df.head()

#feature_list = ['Gender','Age','Marital_Status','NumMonths','Savings_<500','Savings_none','Dependents','Property_rent','Job_management/self-emp/officer/highly qualif emp','Debtors_guarantor','Purpose_CarNew',                           'Purpose_furniture/equip','CreditHistory_none/paid','Purpose_CarUsed','CreditAmount','CreditStatus']
feature_list=['Gender','Age','Marital_Status','NumMonths','Savings_<500','Savings_none','Dependents','Property_rent',
                           'Job_management/self-emp/officer/highly qualif emp','Debtors_guarantor','Purpose_CarNew',
                           'Purpose_furniture/equip','CreditHistory_none/paid','Purpose_CarUsed','CreditAmount',
                           'Collateral_real estate','Debtors_none','Job_unemp/unskilled-non resident','Purpose_others',             
                            'CreditHistory_other','PayBackPercent','Collateral_unknown/none','Purpose_education', 'CreditStatus']

X = German_df.iloc[:, :-1]
y = German_df['CreditStatus']
X.head()
y.head()
German_df.head()

German_df.columns=feature_list
German_df.head()
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

## <font color='crimson'> Building ML model  </font>


### <font color='blue'>1. RANDOM FOREST CLASSIFIER MODEL</font>

#Seting the Hyper Parameters
param_grid = {"max_depth": [3,5,7, 10,None],
              "n_estimators":[3,5,10,25,50,150],
              "max_features": [4,7,15,20]}
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
#Creating the classifier
rf_model = RandomForestClassifier(random_state=40)
grid_search = GridSearchCV(rf_model, param_grid=param_grid, cv=5, scoring='recall', verbose=0)
model_rf = grid_search

mdl_rf = model_rf.fit(data_orig_train.features, data_orig_train.labels.ravel())

from sklearn.metrics import confusion_matrix
conf_mat_rf = confusion_matrix(data_orig_test.labels.ravel(), model_rf.predict(data_orig_test.features))
conf_mat_rf
from sklearn.metrics import accuracy_score
print(accuracy_score(data_orig_test.labels.ravel(), model_rf.predict(data_orig_test.features)))

unique, counts = np.unique(data_orig_test.labels.ravel(), return_counts=True)
dict(zip(unique, counts))


import pandas as pd
import csv
import os
import numpy as np
import sys
from aif360.metrics import *
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
plot_model_performance(mdl_rf, X_test, y_test)

fair_rf = get_fair_metrics_and_plot(  data_orig_test, mdl_rf)
fair_rf

type(data_orig_train)

### <font color='green'>PRE PROCESSING</font>

### Reweighing
from aif360.algorithms.preprocessing import Reweighing

RW_rf = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)

data_transf_train_rf_rw = RW_rf.fit_transform(data_orig_train)
#train and save model
rf_transf_rw = model_rf.fit(data_transf_train_rf_rw.features,
                     data_transf_train_rf_rw.labels.ravel())

data_transf_test_rf_rw = RW_rf.transform(data_orig_test)
fair_rf_rw = get_fair_metrics_and_plot(  data_transf_test_rf_rw, rf_transf_rw, plot=False)

metric_transf_train = BinaryLabelDatasetMetric(data_transf_train_rf_rw, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)
print("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())


fair_rf_rw

from aif360.algorithms.preprocessing import DisparateImpactRemover

DIR_rf = DisparateImpactRemover()
data_transf_train_rf_dir = DIR_rf.fit_transform(data_orig_train)

# Train and save the model
rf_transf_dir = model_rf.fit(data_transf_train_rf_dir.features,data_transf_train_rf_dir.labels.ravel())

fair_dir_rf_dir = get_fair_metrics_and_plot( data_orig_test, rf_transf_dir, plot=False)
fair_dir_rf_dir

#!pip install --user --upgrade tensorflow==1.15.0

#%tensorflow_version 1.15
import tensorflow  as tf
#from tensorflow.compat.v1 import variable_scope
print('Using TensorFlow version', tf.__version__)

#sess = tf.compat.v1.Session()
#import tensorflow as tf

sess = tf.compat.v1.Session()

#import tensorflow as tf
#sess = tf.Session()
tf.compat.v1.reset_default_graph()

from aif360.algorithms.inprocessing.adversarial_debiasing import AdversarialDebiasing
#with tf.variable_scope('debiased_classifier',reuse=tf.AUTO_REUSE):
with tf.compat.v1.Session() as sess:
    with tf.variable_scope('scope1',reuse=tf.AUTO_REUSE) as scope:
        debiased_model_rf_ad = AdversarialDebiasing(privileged_groups = privileged_groups,
                          unprivileged_groups = unprivileged_groups,
                          scope_name=scope,
                          num_epochs=10,
                          debias=True,
                          sess=sess)
#train and save the model
        debiased_model_rf_ad.fit(data_orig_train)
        fair_rf_ad = get_fair_metrics_and_plot(  data_orig_test, debiased_model_rf_ad, plot=False, model_aif=True)

fair_rf_ad

from aif360.algorithms.inprocessing import PrejudiceRemover
debiased_model_pr_rf = PrejudiceRemover()

# Train and save the model
debiased_model_pr_rf.fit(data_orig_train)

fair_rf_pr = get_fair_metrics_and_plot(  data_orig_test, debiased_model_pr_rf, plot=False, model_aif=True)
fair_rf_pr

#######

y_pred = debiased_model_pr_rf.predict(data_orig_test)


data_orig_test_pred = data_orig_test.copy(deepcopy=True)

# Prediction with the original RandomForest model
scores = np.zeros_like(data_orig_test.labels)
scores = mdl_rf.predict_proba(data_orig_test.features)[:,1].reshape(-1,1)
data_orig_test_pred.scores = scores

preds = np.zeros_like(data_orig_test.labels)
preds = mdl_rf.predict(data_orig_test.features).reshape(-1,1)
data_orig_test_pred.labels = preds

def format_probs(probs1):
    probs1 = np.array(probs1)
    probs0 = np.array(1-probs1)
    return np.concatenate((probs0, probs1), axis=1)

### <font color='green'>POST PROCESSING</font>

from aif360.algorithms.postprocessing import EqOddsPostprocessing
EOPP_rf = EqOddsPostprocessing(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups,
                             seed=40)
EOPP_rf = EOPP_rf.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred_rf_eopp = EOPP_rf.predict(data_orig_test_pred)
fair_rf_eo = fair_metrics(  data_orig_test, data_transf_test_pred_rf_eopp, pred_is_dataset=True)

fair_rf_eo

from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing
cost_constraint = "fnr"
CPP_rf = CalibratedEqOddsPostprocessing(privileged_groups = privileged_groups,
                                     unprivileged_groups = unprivileged_groups,
                                     cost_constraint=cost_constraint,
                                     seed=42)

CPP_rf = CPP_rf.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred_rf_cpp = CPP_rf.predict(data_orig_test_pred)
fair_rf_ceo = fair_metrics(  data_orig_test, data_transf_test_pred_rf_cpp, pred_is_dataset=True)

fair_rf_ceo

from aif360.algorithms.postprocessing import RejectOptionClassification
ROC_rf = RejectOptionClassification(privileged_groups = privileged_groups,
                             unprivileged_groups = unprivileged_groups)

ROC_rf = ROC_rf.fit(data_orig_test, data_orig_test_pred)
data_transf_test_pred_rf_roc = ROC_rf.predict(data_orig_test_pred)
fair_rf_roc = fair_metrics(  data_orig_test, data_transf_test_pred_rf_roc, pred_is_dataset=True)
print('SUCCESS: completed 1 model.')

fair_rf_roc

import pickle
pickle.dump(rf_transf_dir,open('dir_age_debiased.pkl','wb'))
dir_model_age=pickle.load(open('dir_age_debiased.pkl','rb'))
data_orig_test.labels[0].ravel()
pred=rf_transf_dir.predict([[ "1"        ,  "1"        ,  "0"        , "24"        ,  "1"        ,    "0"        ,  "1"        ,  "0"        ,  "0"        ,  "0"        ,        "0"        ,  "0"        ,  "0"        ,  "0"        ,  "0.04258831",   "1"        ,  "1"        ,  "0"        ,  "0"        ,  "0"        ,        "4"        ,  "0"        ,  "0"    ]] )
