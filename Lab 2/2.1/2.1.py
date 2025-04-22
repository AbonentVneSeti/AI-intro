from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data() -> pd.DataFrame:
    iris = datasets.load_iris()
    data = pd.DataFrame(iris.data, columns = iris.feature_names)
    data['target'] = iris.target

    return data

def research(data : pd.DataFrame):
    setosa = data.drop(data[data['target'] != 0].index)
    versicolor = data.drop(data[data['target'] != 1].index)
    virginica = data.drop(data[data['target'] != 2].index)


    plt.figure(figsize=(16,8))
    plt.subplots_adjust(hspace=0.3)

    plt.subplot(1,2,1)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.scatter(x = setosa['sepal length (cm)'],y = setosa['sepal width (cm)'], label = 'setosa')#, color = 'blue')
    plt.scatter(x = versicolor['sepal length (cm)'], y = versicolor['sepal width (cm)'], label = 'versicolor')#, color='orange')
    plt.scatter(x = virginica['sepal length (cm)'], y = virginica['sepal width (cm)'], label = 'virginica')#, color='green')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.scatter(x=setosa['petal length (cm)'], y=setosa['petal width (cm)'], label = 'setosa')#, color='blue')
    plt.scatter(x=versicolor['petal length (cm)'], y=versicolor['petal width (cm)'], label = 'versicolor')#, color='orange')
    plt.scatter(x=virginica['petal length (cm)'], y=virginica['petal width (cm)'], label = 'virginica')#, color='green')
    plt.legend()

    plt.tight_layout()
    plt.show()

    sns.pairplot(data, hue='target')
    plt.show()

def sklearn_log_regression(data, label):
    y = data['target']
    x = data.drop('target',axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train,y_train)

    predict = clf.predict(x_test)

    print(f"Модель sklearn, {label}")
    print(f"Точность модели: {clf.score(x_test,y_test)}")

def random_data(samples,features,):
    x,y = datasets.make_classification(n_samples=samples,n_features=features,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    plt.figure(figsize=(8, 6))
    plt.scatter(x[:, 0], x[:, 1], c=y, cmap='bwr', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Сгенерированный датасет для бинарной классификации')
    plt.show()

    clf = LogisticRegression(random_state=0)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)

    print(f"Модель sklearn, random dataset")
    print(f"Точность модели: {clf.score(x_test, y_test)}")

def main():
    data = load_data()

    research(data)

    data_setosa_and_versicolor = data.drop(data[data['target'] == 2].index)
    data_versicolor_and_virginica = data.drop(data[data['target'] == 0].index)

    sklearn_log_regression(data = data_setosa_and_versicolor, label = 'setosa and versicolor')
    sklearn_log_regression(data = data_versicolor_and_virginica, label ='versicolor and virginica')

    random_data(1000,2)

if __name__ == "__main__":
    main()

