import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import *
from sklearn.datasets import load_svmlight_file
from sklearn import svm
import matplotlib.pyplot as plt
from scipy.sparse import hstack, vstack
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
from sklearn.model_selection import GridSearchCV
import csv
import time

#Save prediction vector in Kaggle CSV format
#Input must be a Nx1, 1XN, or N long numpy vector
def kaggleize(predictions,file):

        if(len(predictions.shape)==1):
                predictions.shape = [predictions.shape[0],1]

        ids = 1 + np.arange(predictions.shape[0])[None].T
        kaggle_predictions = np.hstack((ids,predictions)).astype(int)
        writer = csv.writer(open(file, 'w'))
        writer.writerow(['# id','Prediction'])
        writer.writerows(kaggle_predictions)

#### Load Data and create feature & label variables for trian/test/kaggle
train = load_svmlight_file('../../Data/HW2.train.txt')
test = load_svmlight_file('../../Data/HW2.test.txt')
kaggle = load_svmlight_file('../../Data/HW2.kaggle.txt', multilabel=True)

X_train = train[0]
y_train = train[1]

X_test = test[0]
y_test = test[1]

X_kaggle = kaggle[0]
y_kaggle = kaggle[1]


############################# Problem 3 ############################
# Evaluation function for each model
def evaluate(model, X_test, y_test):
    accuracy = model.score(X_test, y_test)*100
    print('Model Performance')
    print('Accuracy = {:0.4f}%.'.format(accuracy))
    return accuracy

# Random Forest default model evaluation
clf = RandomForestClassifier()

t = time.time()
clf = clf.fit(X_train, y_train)
train_time = time.time() - t

t = time.time()
test_pred = clf.predict(X_test)
pred_time = time.time() - t

score = clf.score(X_test, y_test)
kaggle_pred = clf.predict(X_kaggle)
kaggleize(kaggle_pred, 'submission_RF.csv')
print('Random Forest Training Time = {:0.4f}.'.format(train_time))
print('Random Forest Prediction Time = {:0.4f}.'.format(pred_time))
print('Random Forest Accuracy = {:0.4f}.'.format(score))



# Gradient Boost default model evaluation
clf = GradientBoostingClassifier()

t = time.time()
clf = clf.fit(X_train, y_train)
train_time = time.time() - t

t = time.time()
test_pred = clf.predict(X_test)
pred_time = time.time() - t

score = clf.score(X_test, y_test)
kaggle_pred = clf.predict(X_kaggle)
kaggleize(kaggle_pred, 'submission_RF.csv')
print('Gradient Boost Training Time = {:0.4f}.'.format(train_time))
print('Gradient Boost Prediction Time = {:0.4f}.'.format(pred_time))
print('Gradient Boost Accuracy = {:0.4f}.'.format(score))


# SVC default model evaluation
clf = svm.SVC()

t = time.time()
clf = clf.fit(X_train, y_train)
train_time = time.time() - t

t = time.time()
test_pred = clf.predict(X_test)
pred_time = time.time() - t

score = clf.score(X_test, y_test)
kaggle_pred = clf.predict(X_kaggle)
kaggleize(kaggle_pred, 'submission_RF.csv')
print('SVC Training Time = {:0.4f}.'.format(train_time))
print('SVC Prediction Time = {:0.4f}.'.format(pred_time))
print('SVC Accuracy = {:0.4f}.'.format(score))


############################# Problem 4 ############################
# Use the random grid to search for best hyperparameters within wider ranges
learning_rate = [0.1, 0.2, 0.3]  # Learning rate
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)] # Number of trees in random forest
max_features = [10, 20, 30] # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)] # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10] # Minimum number of samples required to split a node  
min_samples_leaf = [2, 5, 10] # Minimum number of samples required at each leaf node  

# First create the base model to tune
clf = GradientBoostingClassifier()

# Random search of parameters, using 5 fold cross validation,
# search across 150 different combinations, and use all available cores
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 150, cv = 5, verbose=2, random_state=2018, n_jobs = -1)

# Fit the random search model              
clf_random.fit(X_train, y_train)

pprint(clf_random.best_params_)
best_param = clf_random.best_params_
random_param = clf_random.cv_results_['params']

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate': learning_rate}

# Save parameters and scores in file
cv0_score = clf_random.cv_results_['split0_test_score']
cv1_score = clf_random.cv_results_['split1_test_score']
cv2_score = clf_random.cv_results_['split2_test_score']
cv3_score = clf_random.cv_results_['split3_test_score']
cv4_score = clf_random.cv_results_['split4_test_score']

filename = "random_cv_pram.csv"
open(filename, 'w', newline='')
with open(filename, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(random_param)
filename = "random_cv_score.csv"
open(filename, 'w', newline='')
with open(filename, 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(cv0_score)
    wr.writerow(cv1_score)
    wr.writerow(cv2_score)
    wr.writerow(cv3_score)
    wr.writerow(cv4_score)

# Best model out of random search
best_rnd_model = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 10,
 max_features = 30,
 min_samples_leaf = 4,
 min_samples_split = 10,
 n_estimators = 444, random_state = 2018)
best_rnd_model.fit(X_train, y_train)
best_train_accuracy = evaluate(best_rnd_model, X_train, y_train)
best_test_accuracy = evaluate(best_rnd_model, X_test, y_test)
print('Gradient Boost Training Accuracy using random search = {:0.4f}.'.format(best_train_accuracy))
print('Gradient Boost Test Accuracy using random search = {:0.4f}.'.format(best_test_accuracy))
##############################
# Create the parameter grid based on the results of random search 
param_grid = {
    'learning_rate': [0.05, 0.1, 0.15],
    'max_depth': [5, 10, 15],
    'max_features': [20, 30, 40],
    'min_samples_leaf': [2, 6, 10],
    'min_samples_split': [4, 7, 10],
    'n_estimators': [400, 600, 800]
}
# Create a based model
gbc = GradientBoostingClassifier()

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = gbc, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(X_train, y_train)

pprint(grid_search.best_params_)

# Here is the best model out of the grid search
best_model = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 5,
 max_features = 30,
 min_samples_leaf = 6,
 min_samples_split = 7,
 n_estimators = 800, random_state = 2018)
best_model.fit(X_train, y_train)
best_train_accuracy = evaluate(best_model, X_train, y_train)
best_test_accuracy = evaluate(best_model, X_test, y_test)
kaggle_pred = best_model.predict(X_kaggle)
kaggleize(kaggle_pred, 'submission_best_model.csv')
print('Gradient Boost Training Accuracy using grid search = {:0.4f}.'.format(best_train_accuracy))
print('Gradient Boost Test Accuracy using grid search = {:0.4f}.'.format(best_test_accuracy))


########### Final Submission ###########
# In order to get the best model, I combined both train and test data to train the model using the optimal hyperparameters. Then predict the Kaggle data.
X_dataset = vstack((X_train, X_test))
y_dataset = hstack((y_train, y_test)).toarray().reshape(-1, )

# Train using both train & test data. Use kaggle data for test error
best_model = GradientBoostingClassifier(learning_rate = 0.1, max_depth = 10,
 max_features = 30,
 min_samples_leaf = 6,
 min_samples_split = 7,
 n_estimators = 800, random_state = 2018)
best_model.fit(X_dataset, y_dataset)
#best_accuracy = evaluate(best_model, X_test_pol, y_test)
kaggle_pred = best_model.predict(X_kaggle)
kaggleize(kaggle_pred, 'submission_both.csv')
