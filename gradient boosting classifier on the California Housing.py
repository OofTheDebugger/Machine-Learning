import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

# Load the California Housing dataset
X, y = fetch_california_housing(return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Train a gradient boosting classifier
est = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                max_depth=1, random_state=0, loss='ls')
est.fit(X_train, y_train)

# Evaluate the model on the test set
mse = np.mean((est.predict(X_test) - y_test) ** 2)
print("Test set MSE: {:.2f}".format(mse))
