import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class RosenblattPerceptron:
    def __init__(self, random_weights=True, n_epochs=10):
        self.random_weights = random_weights
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape

        if self.random_weights:
            self.weights = np.random.rand(n_features)
            self.bias = np.random.rand()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0

        for epoch in range(self.n_epochs):
            for idx, x_i in enumerate(X):
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.step_function(linear_output)

                update = (y[idx] - y_pred)
                self.weights += update * x_i
                self.bias += update

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.step_function(linear_output)
        return y_pred


def main():
    n_samples = 500
    data, labels = make_blobs(n_samples=n_samples,
                              centers=([1.1, 3], [4.5, 6.9]),
                              cluster_std=1.3,
                              random_state=0)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=1)

    # Визуализация исходных данных
    colours = ('green', 'orange')
    plt.figure(figsize=(12,6))
    for n_class in range(2):
        plt.scatter(X_train[y_train == n_class][:, 0],
                   X_train[y_train == n_class][:, 1],
                   c=colours[n_class],
                   edgecolors= 'k',
                   s=20,
                   label='Обучающие данные ' + str(n_class))

        plt.scatter(X_test[y_test == n_class][:, 0],
                    X_test[y_test == n_class][:, 1],
                    c=colours[n_class],
                    s=50,
                    marker='x',
                    label='Тестовые данные ' + str(n_class))

    plt.title("Исходные данные")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.show()

    # Создание и обучение нейронов
    perceptron = RosenblattPerceptron(random_weights=True, n_epochs=10)
    perceptron.fit(X_train, y_train)

    test_predictions = perceptron.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Точность на тестовой выборке: {test_accuracy:.4f}")

    # Визуализация границы
    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(12,6))
    plt.contourf(xx, yy, Z, alpha=0.4)

    for n_class in range(2):
        plt.scatter(X_train[y_train == n_class][:, 0],
                    X_train[y_train == n_class][:, 1],
                    c=colours[n_class],
                    edgecolor='k',
                    s=20,
                    label='Обучающие данные ' + str(n_class))

        plt.scatter(X_test[y_test == n_class][:, 0],
                    X_test[y_test == n_class][:, 1],
                    c=colours[n_class],
                    s=50,
                    marker='x',
                    label='Тестовые данные ' + str(n_class))

    plt.title("Разделяющая граница перцептрона")
    plt.xlabel("Признак 1")
    plt.ylabel("Признак 2")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()