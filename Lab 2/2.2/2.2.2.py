import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap


def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Точность модели: {test_accuracy:.2f}")

    plt.figure(figsize=(10, 6))

    x_min, x_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    y_min, y_max = X[:, 3].min() - 1, X[:, 3].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))

    other_feats = np.array([X[:, 0].mean(), X[:, 1].mean()]).reshape(1, -1)
    mesh_data = np.c_[other_feats.repeat(xx.ravel().shape[0], axis=0),
                      xx.ravel(), yy.ravel()]

    Z = model.predict(mesh_data)
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    plt.scatter(X_train[:, 2], X_train[:, 3], c=y_train, cmap=cmap_bold,
                edgecolor='k', s=20, label='Обучающие данные')
    plt.scatter(X_test[:, 2], X_test[:, 3], c=y_test, cmap=cmap_bold,
                s=50, marker='x', label='Тестовые данные')

    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.title('Многоклассовая логистическая регрессия')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()