import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets


class LinearRegression:
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
			y_pred = np.dot(X, self.weights) + self.bias
			dw = 1 / num_sample * np.dot(X.T, y_pred - y.reshape(-1, 1))
			db = 1 / num_sample * np.sum(y_pred - y)
			if np.max(np.abs(dw)) < 0.0001 and np.max(np.abs(db)) < 0.0001:
				break
			self.weights -= self.lr * dw
			self.bias -= self.lr * db
			# step += 1

	def predict(self, X):
		return np.dot(X, self.weights) + self.bias


def linear_regression(x, y, lr=0.001, terminal_diff=0.0001, iteration=None):
	# get feature num
	f = x.shape[1]

	# initialize weights and bias
	w = []
	for i in range(f):
		w.append(- 1 + 2 * np.random.rand())
	w = np.array(w).astype('float32').reshape(-1, 1)
	b = - 1 + 2 * np.random.rand()

	# loop to update w and b
	itr = 0
	while True:
		if itr % 1000 == 0:
			print('Iteration: %d' % itr)
		temp_w = deepcopy(w)
		temp_b = deepcopy(b)
		for j in range(x.shape[0]):
			pred = np.dot(w.reshape(-1), x[j]) + b
			for k in range(f):
				w[k] += 2 / x.shape[0] * lr * (y[j] - pred) * x[j][k]
			b += 2 / x.shape[0] * lr * (y[j] - pred)
		itr += 1
		if max(abs(temp_w - w)) < terminal_diff and abs(temp_b - b) < terminal_diff:
			w = deepcopy(temp_w)
			b = deepcopy(temp_b)
			print(itr)
			break
	return w, b


if __name__ == '__main__':
	X, y = datasets.make_regression(n_samples=100, n_features=3, noise=20, random_state=4)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	r = LinearRegression()
	r.fit(X_train, y_train)
	preds = r.predict(X_test)
	print('MSE:', mean_squared_error(y_test, preds))
	print('R2:', r2_score(y_test, preds))
	# x1 = np.linspace(1, 100, 100) / 5
	# x2 = np.linspace(1, 100, 100) / 2
	# x3 = np.linspace(1, 100, 100) / 8
	# x = np.vstack((x1, x2, x3)).T
	# y = np.linspace(1, 100, 100) + 0.437
	#
	# w, b = linear_regression(x, y)
	# print(w, b)
	# a = LinearRegression().fit(x, y)
	# print(a.score(x, y))
	# print(a.coef_)
	# print((a.intercept_))
	#
	# fig = plt.figure()
	# plt.scatter(x1, y, c='red')
	# y2 = w * x1 + b
	# plt.plot(x1, y2.reshape(-1), c='blue')
	# plt.show()


