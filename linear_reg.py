import numpy as np
import pandas as pd
from sklearn.datasets import make_regression


class MyLineReg():
    def __init__(self, metric=None, n_iter=100, learning_rate=0.5):
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = None
        self.metric = metric
        self.best_score = -1

    def __str__(self):
        return f"MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}"

    def mae(self, y_predict, y, n):
        return np.sum((y_predict - y) ** 2) * 1 / n

    def mse(self, y_predict, y, n):
        return np.sum(abs(y_predict - y)) * 1 / n

    def rmse(self, y_predict, y, n):
        return np.sqrt(np.sum((y_predict - y) ** 2) * 1 / n)

    def r2(self, y_predict, y, n):
        return 1 - np.sum((y_predict - y) ** 2) / np.sum((y_predict - y.mean()) ** 2)

    def mape(self, y_predict, y, n):
        return 100. / n * np.sum(np.abs((y_predict - y) / y.mean()))

    # def log(self, **kwargs):
    #     if not (self.metric is None):
    #         if kwargs['iteration'] == 0:
    #             print(f'start | loss: {kwargs['loss']} | {kwargs['metric_name']}: {kwargs['metric']}')
    #         elif kwargs['iteration'] % kwargs['verbose'] == 0 or kwargs['iteration'] + 1 == self.n_iter:
    #             print(f'{kwargs['iteration']} | loss: {kwargs['loss']} | {kwargs['metric_name']}: {kwargs['metric']}')
    #     else:
    #         if kwargs['iteration'] == 0:
    #             print(f'start | loss: {kwargs['loss']}')
    #         elif kwargs['iteration'] % kwargs['verbose'] == 0 or kwargs['iteration'] + 1 == self.n_iter:
    #             print(f'{kwargs['iteration']} | loss: {kwargs['loss']}')

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose=0):
        n, m = X.shape
        one_col = pd.Series(np.ones(n))
        X = pd.concat([one_col, X], axis=1)
        metric = None
        match self.metric:
            case 'mae':
                metric = self.mae
            case 'mse':
                metric = self.mse
            case 'rmse':
                metric = self.rmse
            case 'r2':
                metric = self.r2
            case 'mape':
                metric = self.mape

        self.weights = np.ones(m + 1)
        for i in range(self.n_iter):
            y_predict = X @ self.weights  # np.dot(X, self.weights)
            # mse = np.sum((y_predict - y)**2) * 1 / n
            self.best_score = metric(y_predict, y, n)
            # if score < self.best_score or self.best_score == -1:
            #     self.best_score = score
            grt = 2 / n * (y_predict - y) @ X
            self.weights -= self.learning_rate * grt
            # if verbose:
            #     self.log(iteration=i, metric=self.best_score,
            #              verbose=verbose, metric_name=self.metric,
            #              loss=self.mse(y_predict, y, n))

    def predict(self, X: pd.DataFrame):
        one_col = pd.Series(np.ones(X.shape[0]))
        X = pd.concat([one_col, X], axis=1)
        return sum(X @ self.weights)

    def get_coef(self):
        return self.weights[1:]

    def get_best_score(self):
        return self.best_score


if __name__ == "__main__":
    X, y = make_regression(n_samples=400, n_features=14, n_informative=5, noise=5)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    X.columns = [f'col_{col}' for col in X.columns]
    a = MyLineReg('mae')
    a.fit(X, y, verbose=5)
    print(a.get_best_score())
