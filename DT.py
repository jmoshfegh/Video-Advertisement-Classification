# DT.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor
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

def compute_DT(train_x, train_y, test_x):
        # Different values of max_depth to run
        depth = [3, 6, 9, 12, 15]
        # Initialize Variables
        train_score = [0] * len(depth)
        test_score = [0] * len(depth)
        fit_time = [0] * len(depth)
        score_time = [0] * len(depth)

	# make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)

        # indexing
        index = 0
        for d in depth:
                # Create the model
                regr = DecisionTreeRegressor(criterion = "mae", max_depth = d)
                # Cross Validation
                cv_score = cross_validate(regr, train_x, train_y, 
					  return_train_score=True, 
					  scoring = MAE, cv = 5)
                # Extract Statistics, scorer negates compute_error output
                train_score[index] = -cv_score['train_score'].mean()
                test_score[index] = -cv_score['test_score'].mean()
                fit_time[index] = 1000 * cv_score['fit_time'].sum()
                score_time[index] = 1000 * cv_score['score_time'].sum()

                # Print Statistics
                print('Depth of Decision Tree:', d)
                print('train score', train_score[index])
                print('test score', test_score[index])
                print('fit time', fit_time[index])
                print('score time', score_time[index])
                print('===============================')

                # Fit Model
                regr.fit(train_x, train_y)
                
                # Make Prediction
                test_y = regr.predict(test_x)
                
                # Create test output values
                predicted_y = test_y * -1
                # Output file location
                file_name = '../Predictions/Decision_Tree_depth_%d.csv' % d
                # Writing output in Kaggle format
                print('Writing output to ', file_name)
                kaggle.kaggleize(predicted_y, file_name)

                # Increase indexing
                index = index + 1

        # Plot CV Time
        plt.figure(num=None, figsize=(16, 8), dpi=80, facecolor='w', edgecolor='k')
        plt.plot(depth, fit_time, '.')
        plt.xlabel('Depth of Decision Tree')
        plt.ylabel('Cross Validation Time [msec]')
        plt.savefig('../Figures/DT_cv_time.png')
