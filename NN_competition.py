# NN_competition.py module
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import kaggle
from sklearn.metrics.scorer import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from keras import models
from keras import layers
from keras.layers import Dropout
from keras import optimizers

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

# Create function returning a compiled network
def create_network():
    
    # Start neural network
    network = models.Sequential()

    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=20, activation='relu', input_shape=(52,)))
    network.add(Dropout(0.1))
    # Add fully connected layer with a ReLU activation function
    network.add(layers.Dense(units=20, activation='relu'))
    network.add(Dropout(0.1))
    # Add fully connected layer with a sigmoid activation function
    network.add(layers.Dense(units=1, activation='relu'))

    opti_adam=optimizers.Adam(lr=0.1, beta_1=0.9)
    # Compile neural network
    network.compile(loss='MAE', # Cross-entropy
                    optimizer=opti_adam, # Root Mean Square Propagation
                    metrics=['accuracy']) # Accuracy performance metric
    
    # Return compiled network
    return network


def compute_NN(train_x, train_y, test_x):

        np.random.seed(2018)
        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)

        scaler = StandardScaler()
        
        ######### Neural Network #########
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        neural_network = KerasRegressor(build_fn=create_network, 
                                 epochs=100, 
                                 batch_size=100, 
                                 verbose=0)

        scores = cross_val_score(neural_network, train_x, train_y, scoring=MAE, cv=5)
        print('mean scores', -np.mean(scores))
        

def final_model(train_x, train_y, test_x):
        np.random.seed(2018)
        # make MAE scoring
        MAE = make_scorer(compute_error, greater_is_better = False)
        
        scaler = StandardScaler()
        scaler.fit(train_x)
        train_x = scaler.transform(train_x)
        test_x = scaler.transform(test_x)

        # create model
        model = Sequential()
        model.add(Dense(20, input_shape=(52,), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(20, activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation='relu'))

        opti_adam=optimizers.Adam(lr=0.1, beta_1=0.9)
        # Compile model
        model.compile(loss='MAE', optimizer=opti_adam, metrics=['accuracy'])
        # Fit the model
        model.fit(train_x, train_y, epochs=150, batch_size=200)
        # evaluate the model
        scores = model.evaluate(train_x, train_y)
        print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

        test_y = model.predict(test_x)

        predicted_y = test_y * -1
        # Output file location
        file_name = '../Predictions/NN_best_competition.csv'

        # Writing output in Kaggle format
        print('Writing output to ', file_name)
        kaggle.kaggleize(predicted_y, file_name)
