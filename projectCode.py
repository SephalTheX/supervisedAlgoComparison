# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 12:32:47 2020

@author: aparn
"""

'''
HELPFUL LIBRARIES
numpy
scikit - this is where most of the things you need to implement is available
       - StandardScaler(), KFold(), SVM() ..... all models
       - auc(), fscore(), confusion_matrix()
matplotlib - plot() etc...
'''

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import validation_curve
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

isolet_tuple = pd.read_csv("./dataset/ISOLET/isolet1+2+3+4.data", header=None, delim_whitespace=False)
isolet = pd.DataFrame(isolet_tuple)

isolet_features = isolet.drop(isolet.columns[[len(isolet.iloc[0]) - 1]], axis=1)
isolet_labels = isolet.iloc[:, -1:]

letters_tuple = pd.read_csv("./dataset/Letter Recognition/letter-recognition.data")
letters_tuple.columns = ["letter", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar",
                         "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"]
letters = pd.DataFrame(letters_tuple)
letters_features = letters.drop(letters.columns[[0]], axis=1)
letters_label = letters.iloc[:, 0]

sens_less = pd.read_csv("./dataset/Sensorless Drive Diagnosis/Sensorless_drive_diagnosis.txt", sep=" ")
sens_features = sens_less.drop(sens_less.columns[[len(sens_less.iloc[0]) - 1]], axis=1)
sens_labels = sens_less.iloc[:, -1:]

from sklearn.model_selection import RepeatedKFold

clfRF = RandomForestClassifier()

clfSVM = SVC(random_state=12345)

clfReg = LogisticRegression(multi_class='multinomial',
                            solver='newton-cg',
                            random_state=12345)
pipe1 = Pipeline([('std', StandardScaler()),
                  ('classifier', clfRF)])

pipe2 = Pipeline([('std', StandardScaler()),
                  ('classifier', clfSVM)])

pipe3 = Pipeline([('std', StandardScaler()),
                  ('classifier', clfReg)])

# Create search space of candidate learning algorithms and their hyperparameters
param_grid_RF = [{'classifier': [RandomForestClassifier()],
                  'classifier__n_estimators': [10, 100, 1000],
                  'classifier__max_features': [1, 2, 3]}]
param_grid_svm = [{'classifier__kernel': ['rbf'],
                   'classifier__C': np.power(10., np.arange(-4, 4)),
                   'classifier__gamma': np.power(10., np.arange(-5, 0))},
                  {'classifier__kernel': ['linear'],
                   'classifier__C': np.power(10., np.arange(-4, 4))},
                  {'classifier__kernel': ['sigmoid'],
                   'classifier__C': np.power(10., np.arange(-4, 4))},
                  ]

param_grid_logistic = [{'classifier__penalty': ['l2'],
                        'classifier__C': np.power(10., np.arange(-4, 4))},
                       {'classifier__penalty': ['l1'],
                        'classifier__C': np.power(10., np.arange(-4, 4))}]
# clf = GridSearchCV(pipe, search_space, cv=RepeatedKFold(n_splits=2, n_repeats=3, random_state=123), verbose=0)
# Change the number of splits to 10 after done testing

import warnings

# there are a lot of convergence warnings for some params, however be careful with this!!
# sometimes you need to see those wanrings, and now we've screwed tha tup for the whole notebook from here on!!
warnings.filterwarnings('ignore')
griddles = {}  # Yummy
# isolet_train_x, isolet_test_x, isolet_train_y, isolet_test_y = train_test_split(isolet_features[:150], isolet_labels[:150], test_size=0.2, random_state=30)
isolet_train_x, isolet_test_x, isolet_train_y, isolet_test_y = train_test_split(isolet_features, isolet_labels,
                                                                                train_size=0.8, random_state=12345,
                                                                                stratify=isolet_labels)
letters_train_x, letters_test_x, letters_train_y, letters_test_y = train_test_split(
    letters_features[:int(len(letters_features) / 2)], letters_label[:int(len(letters_features) / 2)], train_size=0.8,
    random_state=12345, stratify=letters_label[:int(len(letters_features) / 2)])
sens_train_x, sens_test_x, sens_train_y, sens_test_y = train_test_split(sens_features[:int(len(sens_features) / 2)],
                                                                        sens_labels[:int(len(sens_features) / 2)],
                                                                        train_size=0.8, random_state=12345,
                                                                        stratify=sens_labels[
                                                                                 :int(len(sens_features) / 2)])

for param_g, estimates, names in zip((param_grid_RF, param_grid_svm, param_grid_logistic), (pipe1, pipe2, pipe3),
                                     ('RF', 'SVM', 'LR')):
    gc = GridSearchCV(estimator=estimates, param_grid=param_g, scoring='accuracy', n_jobs=1, cv=2, verbose=0,
                      refit=True)
    griddles[names] = gc

cv_scores = {name: [] for name, gs_est in griddles.items()}
# clf = GridSearchCV(pipe, search_space, cv=StratifiedKFold(n_splits=10), verbose=0)
# skfolded = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)
# best_model = clf.fit(isolet_train_x, isolet_train_y)
skfolded = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
c = 1

import warnings

# there are a lot of convergence warnings for some params, however be careful with this!!
# sometimes you need to see those wanrings, and now we've screwed tha tup for the whole notebook from here on!!
warnings.filterwarnings('ignore')

# best = griddles['LR'].fit(isolet_train_x, isolet_train_y)


for i, j, k in ([(letters_train_x, letters_train_y, 'letters'), (isolet_train_x, isolet_train_y, 'isolet'),
                 (sens_train_x, sens_train_y, 'sens')]):
    for outer_tr_ind, outer_val_ind in skfolded.split(i, j):
        for name, gs_est in sorted(griddles.items()):
            # print(j)
            print('dataset:%-8s outer fold %d/5 | tuning %-8s' % (k, c, name))
            # print(isolet_train_x.iloc[outer_tr_ind])
            gs_est.fit(i.iloc[outer_tr_ind], j.iloc[outer_tr_ind])
            y_pred = gs_est.predict(i.iloc[outer_val_ind])
            acc = accuracy_score(y_true=j.iloc[outer_val_ind], y_pred=y_pred)
            print(' | inner Accuracy %.2f%% | outer Accuracy %.2f%%' % (gs_est.best_score_ * 100, acc * 100))
            cv_scores[name].append(acc)
        c += 1
    c = 1


# Looking at the results
for name in cv_scores:
    print('%-8s | outer CV acc. %.2f%% +\- %.3f' % (
        name, 100 * np.mean(cv_scores[name]), 100 * np.std(cv_scores[name])))
print()
for name in cv_scores:
    print('{} best parameters'.format(name), griddles[name].best_params_)

# %%

best_algo = griddles['RF']

best_algo.fit(isolet_train_x, isolet_train_y)
train_acc = accuracy_score(y_true=isolet_train_y, y_pred=best_algo.predict(isolet_train_x))
test_acc = accuracy_score(y_true=isolet_test_y, y_pred=best_algo.predict(isolet_test_x))

print('Accuracy %.2f%% (average over CV test folds)' %
      (100 * best_algo.best_score_))
print('Best Parameters: %s' % griddles['SVM'].best_params_)
print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))

print(best_algo.cv_results_)

# %%

best_algo = griddles['SVM']

best_algo.fit(isolet_train_x, isolet_train_y)
train_acc = accuracy_score(y_true=isolet_train_y, y_pred=best_algo.predict(isolet_train_x))
test_acc = accuracy_score(y_true=isolet_test_y, y_pred=best_algo.predict(isolet_test_x))

print('Accuracy %.2f%% (average over CV test folds)' %
      (100 * best_algo.best_score_))
print('Best Parameters: %s' % griddles['SVM'].best_params_)
print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))
print(best_algo.cv_results_)

# %%

best_algo = griddles['LR']

best_algo.fit(isolet_train_x, isolet_train_y)
train_acc = accuracy_score(y_true=isolet_train_y, y_pred=best_algo.predict(isolet_train_x))
test_acc = accuracy_score(y_true=isolet_test_y, y_pred=best_algo.predict(isolet_test_x))

print('Accuracy %.2f%% (average over CV test folds)' %
      (100 * best_algo.best_score_))
print('Best Parameters: %s' % griddles['SVM'].best_params_)
print('Training Accuracy: %.2f%%' % (100 * train_acc))
print('Test Accuracy: %.2f%%' % (100 * test_acc))
print(best_algo.cv_results_)
