from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from supervised_learning.linear_regression import LinearRegression

max_expected_mse = 30


def test_linear_regression():
    X, y = load_boston(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2021)

    X_train_mean = X_train.mean(axis=0)
    X_train_std = X_train.mean(axis=0)
    X_train_norm = (X_train - X_train_mean) / X_train_std
    X_test_norm = (X_test - X_train_mean) / X_train_std

    model = LinearRegression(learning_rate=0.001, n_iterations=10000, regularization='l2', C=0.5)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)

    assert mean_squared_error(y_test, y_pred) < max_expected_mse
