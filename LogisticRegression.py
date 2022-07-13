import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn import datasets


class LogisticRegression:
	def __init__(self, lr=0.001, n_iters=1000):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None

	def fit(self, X, y):
		num_sample, num_features = X.shape
		self.weights = np.random.randn(num_features, 1)
		self.bias = np.random.rand()
		step = 0
		while step < self.n_iters:
			wx_b = np.dot(X, self.weights) + self.bias
			y_pred = self._sigmoid(wx_b)
			dw = 1 / num_sample * np.dot(X.T, y_pred - y.reshape(-1, 1))
			db = 1 / num_sample * np.sum(y_pred - y)
			if np.max(np.abs(dw)) < 0.0001 and np.max(np.abs(db)) < 0.0001:
				break
			self.weights -= self.lr * dw
			self.bias -= self.lr * db
			step += 1

	def _sigmoid(self, X):
		return 1 / (1 + np.exp(-X))

	def predict(self, X):
		wx_b = np.dot(X, self.weights) + self.bias
		y_pred = self._sigmoid(wx_b)
		y_pred_cls = [1 if i > 0.5 else 0 for i in y_pred]
		return y_pred_cls


if __name__ == '__main__':
	bc = datasets.load_breast_cancer()
	X, y = bc.data, bc.target
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	r = LogisticRegression()
	r.fit(X_train, y_train)
	preds = r.predict(X_test)
	print('Acc:', accuracy_score(y_test, preds))
