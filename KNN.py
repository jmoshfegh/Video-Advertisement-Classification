# KNN.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.neighbors import KNeighborsRegressor
import matplotlib
import matplotlib.pyplot as plt
import kaggle
from sklearn.metrics.scorer import make_scorer

plt.switch_backend('agg')
matplotlib.rcParams.update({'font.size': 22})

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

def compute_KNN(train_x, train_y, test_x):
        # Different number of neighbors to run
        neighbor = [3, 5, 10, 20, 25]
        # Initialize Variables
        train_score = [0] * len(neighbor)
        test_score = [0] * len(neighbor)
        fit_time = [0] * len(neighbor)
        score_time = [0] * len(neighbor)

        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)
        
        # indexing
        index = 0
        for n in neighbor:
                # Create the model
                regr = KNeighborsRegressor(n_neighbors = n)
                # Cross Validation
                cv_score = cross_validate(regr, train_x, train_y,
                                          return_train_score=True,
                                          scoring = MAE, cv = 2)
                # Extract Statistics, scorer negates compute_error output
                train_score[index] = -cv_score['train_score'].mean()
                test_score[index] = -cv_score['test_score'].mean()

                # Print Statistics
                print('Number of Neighbors:', n)
                print('train score', train_score[index])
                print('test score', test_score[index])
                print('===============================')

                # Fit Model
                regr.fit(train_x, train_y)
                
                # Make Prediction
                test_y = regr.predict(test_x)
                
                # Create test output values
                predicted_y = test_y * -1
                
                # Output file location
                file_name = '../Predictions/KNearestN_Neighbors_%d.csv' % n
                
                # Writing output in Kaggle format
                print('Writing output to ', file_name)
                kaggle.kaggleize(predicted_y, file_name)

                # Increase indexing
                index = index + 1

def distance_effect(train_x, train_y, test_x):
	regr = KNeighborsRegressor(n_neighbors = 3, p=1)
	# Fit Model
	regr.fit(train_x, train_y)
	
	# Make Prediction
	test_y = regr.predict(test_x)

	# Create test output values
	predicted_y = test_y * -1

	# Output file location
	file_name = '../Predictions/KNN_Manhattan.csv' 

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(predicted_y, file_name)

	regr = KNeighborsRegressor(n_neighbors = 3, metric='chebyshev')
	# Fit Model
	regr.fit(train_x, train_y)
	
	# Make Prediction
	test_y = regr.predict(test_x)

	# Create test output values
	predicted_y = test_y * -1

	# Output file location
	file_name = '../Predictions/KNN_chebyshev.csv' 

	# Writing output in Kaggle format
	print('Writing output to ', file_name)
	kaggle.kaggleize(predicted_y, file_name)
