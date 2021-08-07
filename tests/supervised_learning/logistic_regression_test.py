from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from supervised_learning.logistic_regression import LogisticRegression, LogisticRegressionV2

min_expected_accuracy = 0.9


def test_logistic_regression_v1():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2021)

    model = LogisticRegression(learning_rate=0.001, n_iterations=4000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).sum() / len(y_test)
    print(accuracy)

    assert accuracy > min_expected_accuracy


def test_logistic_regression_v1_with_regularization():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2021)

    model = LogisticRegression(learning_rate=0.001, n_iterations=4000, regularization='l2', C=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).sum() / len(y_test)
    print(accuracy)

    assert accuracy > min_expected_accuracy


def test_logistic_regression_v2_with_regularization():
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2021)

    model = LogisticRegressionV2(learning_rate=0.001, n_iterations=4000, regularization='l2', C=0.01)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = (y_pred == y_test).sum() / len(y_test)

    assert accuracy > min_expected_accuracy
