import numpy
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class MyLineReg():
    def __init__(self, n_iter=100, learning_rate=0.5):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        n, m = X.shape
        one_col = pd.DataFrame(np.ones(n))
        X = pd.concat([one_col, X], axis=1)
        # print(X)
        self.weights = np.ones(m + 1)
        if verbose:
            for i in range(self.n_iter):
                y_predict = np.dot(X, self.weights)
                mse = np.sum((y_predict - y)**2) * 1 / n
                grt = 2/n * np.dot((y_predict - y), X)
                self.weights -= self.learning_rate * grt
                if i % verbose == 0:
                    print(f'{i} | loss: {mse}')
                elif i == 0:
                    print(f'start | loss: {mse}')

    def get_coef(self):
        return self.weights[1:]


if __name__ == "__main__":
    X, y = make_regression(n_samples=1000, n_features=14, n_informative=10, noise=15, random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]
    a = MyLineReg(n_iter=100, learning_rate=0.1)
    a.fit(X, y, verbose=5)
    print(a.get_coef().mean())
