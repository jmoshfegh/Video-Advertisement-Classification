# SVMR.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import kaggle
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

def compute_SVR(train_x, train_y, test_x):

        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)

        ######### SVR - Polynomial/rbf Kernel #########
        # make pipeline
        std_SVR = make_pipeline(StandardScaler(), SVR())
        params = {'svr__kernel': ['poly', 'rbf'], 'svr__degree': [1, 2]}
        gs = GridSearchCV(estimator = std_SVR, param_grid = params, scoring = MAE, n_jobs=-1, cv = 5, return_train_score = True)


	# fit grid search
        gs.fit(train_x, train_y)

        print('SVR train score', -gs.cv_results_['mean_train_score'])
        print('SVR test score', -gs.cv_results_['mean_test_score'])
        print('Best Parameter', gs.best_params_)
        print('Best score', -gs.best_score_)
        print('Parameters', gs.cv_results_['params'])
        
        # Train the best Model
        best_SVR = make_pipeline(StandardScaler(), SVR(kernel='poly', degree=1))
        best_SVR.fit(train_x, train_y)

        # Make Prediction
        test_y = best_SVR.predict(test_x)
        # Create test output values
        predicted_y = test_y * -1
        # Output file location
        file_name = '../Predictions/SVR_best.csv'

        # Writing output in Kaggle format
        print('Writing output to ', file_name)
        kaggle.kaggleize(predicted_y, file_name)
