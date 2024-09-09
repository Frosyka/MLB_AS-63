import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self, input_size, l_rate=0.0001, epochs=10000):
        self.weights = np.zeros(input_size + 1)
        self.l_rate = l_rate
        self.epochs = epochs

    def activation(self, x):
        return x

    def predict(self, X):
        linear_output = np.dot(X, self.weights[1:]) + self.weights[0]
        return self.activation(linear_output)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            for i in range(len(X)):
                prediction = self.predict(X[i])
                error = y[i] - prediction

                if np.isnan(error) or np.isinf(error):
                    print(f"Ошибка: {error}, Индекс: {i}, Прогноз: {prediction}, Цель: {y[i]}")

                self.weights[1:] += self.l_rate * error * X[i]
                self.weights[0] += self.l_rate * error

                if np.any(np.isnan(self.weights)) or np.any(np.isinf(self.weights)):
                    print(f"Вес: {self.weights}, Индекс: {i}, Прогноз: {prediction}, Цель: {y[i]}")
                    return

X = np.linspace(-10, 10, 100)
y = X**2
X = X.reshape(-1, 1)

X_mean = np.mean(X)
X_std = np.std(X)
X_normalized = (X - X_mean) / X_std

X_poly = np.hstack((X_normalized, X_normalized**2))

perceptron = Perceptron(input_size=2, l_rate=0.0001, epochs=1000)
perceptron.fit(X_poly, y)

X_test = np.linspace(-10, 10, 100).reshape(-1, 1)
X_test_normalized = (X_test - X_mean) / X_std
X_test_poly = np.hstack((X_test_normalized, X_test_normalized**2))
y_pred = perceptron.predict(X_test_poly)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Исходные данные')
plt.plot(X_test, y_pred, color='red', label='Прогноз персептрона')
plt.plot(X_test, X_test**2, color='green', linestyle='dashed', label='Истинная парабола')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Персептрон для параболической функции')
plt.legend()
plt.show()
