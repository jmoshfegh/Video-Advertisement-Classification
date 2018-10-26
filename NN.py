# NN.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import kaggle
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

def compute_NN(train_x, train_y, test_x):

        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)

        ######### Neural Network #########
        # make pipeline
        std_NN = make_pipeline(StandardScaler(), MLPRegressor())
        params = {'mlpregressor__hidden_layer_sizes': [(10,), (20,), (30,), (40,)],
                  'mlpregressor__max_iter': [1000]}
        gs = GridSearchCV(estimator = std_NN, param_grid = params, scoring = MAE, n_jobs=-1, cv = 5, return_train_score = True)

	# fit grid search
        gs.fit(train_x, train_y)

        print('NN train score', -gs.cv_results_['mean_train_score'])
        print('NN test score', -gs.cv_results_['mean_test_score'])
        print('Best Parameter', gs.best_params_)
        print('Best score', -gs.best_score_)
        print('Parameters', gs.cv_results_['params'])
        
        
        # Train the best Model
        best_NN = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(20,)))
        best_NN.fit(train_x, train_y)

        # Make Prediction
        test_y = best_NN.predict(test_x)
        # Create test output values
        predicted_y = test_y * -1
        # Output file location
        file_name = '../Predictions/NN_best.csv'

        # Writing output in Kaggle format
        print('Writing output to ', file_name)
        kaggle.kaggleize(predicted_y, file_name)
