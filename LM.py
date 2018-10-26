# LM.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
import kaggle
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

def compute_LM(train_x, train_y, test_x):
        # Different values of alpha to run
        alpha = [1e-6, 1e-4, 1e-2, 1, 10]
        # Initialize Variables
        train_score = [0] * len(alpha)
        test_score = [0] * len(alpha)

        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)

        ######### Ridge Regression #########
        # make pipeline
        std_Ridge = make_pipeline(StandardScaler(), Ridge())
        #std_Ridge = make_pipeline(Normalizer(), Ridge())
        params = {'ridge__alpha': [1e-6, 1e-4, 1e-2, 1, 10]}
        gs = GridSearchCV(estimator = std_Ridge, param_grid = params, scoring = MAE, cv = 5, return_train_score = True)

	# fit grid search
        gs.fit(train_x, train_y)

        print('Ridge train score', -gs.cv_results_['mean_train_score'])
        print('Ridge test score', -gs.cv_results_['mean_test_score'])

        ######### Lasso Regression #########
        # make pipeline
        std_Lasso = make_pipeline(StandardScaler(), Lasso())
        #std_Lasso = make_pipeline(Normalizer(), Lasso())
        params = {'lasso__alpha': [1e-6, 1e-4, 1e-2, 1, 10], 'lasso__max_iter': [10000]}
        gs = GridSearchCV(estimator = std_Lasso, param_grid = params, scoring = MAE, cv = 5, return_train_score = True)

	# fit grid search
        gs.fit(train_x, train_y)

        print('Lasso train score', -gs.cv_results_['mean_train_score'])
        print('Lasso test score', -gs.cv_results_['mean_test_score'])


        # Train the best Model
        best_Lasso = make_pipeline(StandardScaler(), Lasso(alpha = 1, max_iter = 10000))
        best_Lasso.fit(train_x, train_y)


        print('best Lasso coefficient', best_Lasso.named_steps['lasso'].coef_)

        # Make Prediction
        test_y = best_Lasso.predict(test_x)
        # Create test output values
        predicted_y = test_y * -1
        # Output file location
        file_name = '../Predictions/LM_best.csv'

        # Writing output in Kaggle format
        print('Writing output to ', file_name)
        kaggle.kaggleize(predicted_y, file_name)
