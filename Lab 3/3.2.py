import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron as SkPerceptron
from sklearn.svm import SVC


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
    # Часть 1
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
                   edgecolors='k',
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

    # RosenblattPerceptron
    perceptron = RosenblattPerceptron()
    perceptron.fit(X_train, y_train)

    test_predictions = perceptron.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    print(f"Точность RosenblattPerceptron на случайных данных: {test_accuracy:.4f}")

    # SkPerceptron
    sk_perceptron = SkPerceptron(max_iter=100, random_state=1)
    sk_perceptron.fit(X_train, y_train)
    sk_test_predictions = sk_perceptron.predict(X_test)
    sk_test_accuracy = accuracy_score(y_test, sk_test_predictions)
    print(f"Точность SkPerceptron на случайных данных: {sk_test_accuracy:.4f}")

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

    # Часть 2
    iris = load_iris()
    X_iris = iris.data[-100:]
    y_iris = iris.target[-100:]
    y_iris = y_iris - 1

    X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
        X_iris, y_iris, test_size=0.2, random_state=1)

    # RosenblattPerceptron
    perceptron_iris = RosenblattPerceptron(n_epochs=20)
    perceptron_iris.fit(X_train_iris, y_train_iris)
    iris_predictions = perceptron_iris.predict(X_test_iris)
    iris_accuracy = accuracy_score(y_test_iris, iris_predictions)
    print(f"\nТочность MyPerceptron на Iris: {iris_accuracy:.4f}")

    # SkPerceptron
    sk_perceptron_iris = SkPerceptron(max_iter=100, random_state=1)
    sk_perceptron_iris.fit(X_train_iris, y_train_iris)
    sk_iris_predictions = sk_perceptron_iris.predict(X_test_iris)
    sk_iris_accuracy = accuracy_score(y_test_iris, sk_iris_predictions)
    print(f"Точность SkPerceptron на Iris: {sk_iris_accuracy:.4f}")

    # SVM
    svm = SVC(kernel='linear', random_state=1)
    svm.fit(X_train_iris, y_train_iris)
    svm_predictions = svm.predict(X_test_iris)
    svm_accuracy = accuracy_score(y_test_iris, svm_predictions)
    print(f"Точность SVM на Iris: {svm_accuracy:.4f}")

if __name__ == "__main__":
    main()