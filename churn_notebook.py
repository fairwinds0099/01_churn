import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import json
import pickle
import requests

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import auc, average_precision_score, f1_score, confusion_matrix, roc_auc_score, auc, \
    accuracy_score, log_loss, roc_curve, precision_recall_curve
from sklearn.ensemble import GradientBoostingClassifier

os.chdir('/Users/apple4u/Desktop/data_eng/project_pro/01_churn')


class DataAnalysisTools():

    def get_basic_stats(dfname):
        print("Shape of dataframe is " + str(dfname.shape))
        print("Below are datatypes of columns in DF")
        print(dfname.dtypes.sort_values())
        print("Below are missing values in each column")
        print(dfname.isna().sum().sort_values())
        print("Below are the number of unique values taken by a column")
        print(dfname.nunique().sort_values())
        print("Below are some records in DF")
        print("Below is distribution of numeric variables")
        print(dfname.describe())
        print(dfname.head())

    def cat_to_binary(df, varname):
        df[varname + '_num'] = df[varname].apply(lambda x: 1 if x == 'yes' else 0)
        print("checking")
        print(df.groupby([varname + '_num', varname]).size())
        return df


if __name__ == '__main__':
    # reading the required data files
    trainer = pd.read_csv('Telecom_Train.csv')
    tester = pd.read_csv('Telecom_Test.csv')

    # checking the train to test data ratio
    ratio = tester.shape[0] / trainer.shape[0]
    sum_of_trainer_nulls: object = trainer.isna().sum()

    # DataAnalysisTools.get_basic_stats(trainer)

    trainer = trainer.drop(['Unnamed: 0'], axis=1)
    tester = tester.drop(['Unnamed: 0'], axis=1)

    trainer2 = trainer.copy()
    tester2 = tester.copy()

    # creating binary variables from categorical variables that take just 2 unique values
    yes_no_vars = ['churn', 'international_plan', 'voice_mail_plan']

    for indexer, varname in enumerate(yes_no_vars):
        trainer2 = DataAnalysisTools.cat_to_binary(trainer2, varname)
        tester2 = DataAnalysisTools.cat_to_binary(tester2, varname)

    # dropping object vars that have been converted to numeric
    trainer2 = trainer2.drop(yes_no_vars, axis=1)
    tester2 = tester2.drop(yes_no_vars, axis=1)

    # univariate analysis of categorical variables
    # Visualizing the churn variable
    # creating a list of continuous variables, which would be visualized using boxplot
    continuous_vars = trainer.select_dtypes([np.number]).columns.tolist()

    # univariate analysis of continuous variables

    # Creating a charge per minute variable..in both dataframes
    # Intuitively, we expect customer with high value of this variable to have higher churn rate
    charge_vars = [x for x in trainer.columns if 'charge' in x]
    minutes_vars = [x for x in trainer.columns if 'minutes' in x]
    print(charge_vars)
    print(minutes_vars)


    def create_cpm(df):
        df['total_charges'] = 0
        df['total_minutes'] = 0
        for indexer in range(0, len(charge_vars)):
            df['total_charges'] += df[charge_vars[indexer]]
            df['total_minutes'] += df[minutes_vars[indexer]]
        df['charge_per_minute'] = np.where(df['total_minutes'] > 0, df['total_charges'] / df['total_minutes'], 0)
        df.drop(['total_minutes', 'total_charges'], axis=1, inplace=True)
        print(df['charge_per_minute'].describe())
        return df


    trainer2 = create_cpm(trainer2)
    tester2 = create_cpm(tester2)

    # we have identified 5 variables that can be dropped
    drop_after_corr = ['total_day_charge', 'total_eve_charge', 'total_night_charge', 'total_intl_charge',
                       'voice_mail_plan_num']
    trainer3 = trainer2.drop(drop_after_corr, axis=1)
    tester3 = tester2.drop(drop_after_corr, axis=1)

    # doing ohe of the 2 categorical varables
    cat_columns = ['state', 'area_code']
    trainer3 = pd.concat([trainer3, pd.get_dummies(trainer3[cat_columns], drop_first=True)], axis=1)
    tester3 = pd.concat([tester3, pd.get_dummies(tester3[cat_columns], drop_first=True)], axis=1)
    trainer3 = trainer3.drop(cat_columns, axis=1)
    tester3 = tester3.drop(cat_columns, axis=1)

    ##MODELING
    X_train = trainer3.drop('churn_num', axis=1)
    Y_train = trainer3['churn_num']
    X_test = tester3.drop('churn_num', axis=1)
    Y_test = tester3['churn_num']

    # LR with Hyper Param
    lr = LogisticRegression(random_state=42, solver='liblinear')
    param_gridd = {'penalty': ['l1', 'l2'], 'C': [0.1, 2, 3, 5]}
    CV_lr = GridSearchCV(estimator=lr, param_grid=param_gridd, cv=5)
    CV_lr.fit(X_train, Y_train)
    lr_best = CV_lr.best_estimator_
    test_score_lr = lr_best.predict_proba(X_test)[:, 1]
    pd.Series(test_score_lr).describe()
    # Confusion Matrix Logistic Regression
    cm_lr = confusion_matrix(Y_test, (test_score_lr >= 0.5))
    print(cm_lr)

    # Gradient Boosting  with Hyper Param
    #gbr = GradientBoostingClassifier(random_state=42)
    #param_grid = {'n_estimators': [50, 100, 500], 'max_features': ['auto'], 'learning_rate': [0.01, 0.05, 0.1, 0.2]}
    #CV_gbr = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5)
    #CV_gbr.fit(X_train, Y_train)
    #gbr_best = CV_gbr.best_estimator_
    #test_score_gbr = gbr_best.predict_proba(X_test)[:, 1]
    # Confusion Matrix Gradient Boosting
    #cm_gbr = confusion_matrix(Y_test, (test_score_gbr >= 0.5))
    #print(cm_gbr)

    # ROC for LR and Gradient Boost Classifier
# roc_auc_lr = roc_auc_score(Y_test, test_scrore_lr, average='macro')
# roc_auc_gbr = roc_auc_score(Y_test, test_scrore_gbr, average='macro')

# Print ROC for LR and Gradient Boosting Classfier


# Recursive Feature Elimination

# Model Implementation
model_columns = list(X_train.columns)
pickle.dump(lr_best, open('model.pkl', 'wb'))
pickle.dump(model_columns, open('model_columns.pkl', 'wb'))



