import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Данные загружены.")

    initial_num = len(data)

    for i in ['PassengerId', 'Name', 'Ticket', 'Cabin']:
        data = data.drop(i, axis=1)

    data = data.dropna()

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})


    print(f"Утерянный процент данных: {100*(initial_num - len(data)) / initial_num : .1f}%")

    return data

def sklearn_log_regression(data, label, target):
    y = data[target]
    x = data.drop(target,axis = 1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    clf = LogisticRegression(random_state=0,max_iter=1000)
    clf.fit(x_train,y_train)

    predict = clf.predict(x_test)

    score = clf.score(x_test,y_test)
    print(f"Модель sklearn, {label}")
    print(f"Точность модели: {score}")
    return score

def main():
    data = load_data("Titanic.csv")

    data_without_emb = data.drop('Embarked',axis = 1)

    score_with_emb = sklearn_log_regression(data = data,label = 'with Embarked', target = 'Survived')
    score_without_emb = sklearn_log_regression(data = data_without_emb, label = 'without Embarked', target = 'Survived')

    delta = score_with_emb - score_without_emb
    if delta > 0:
        print(f"Embarked улучшает точность модели на {delta*100 :.1f}%")
    elif delta < 0:
        print(f"Embarked ухудшает точность модели на {-1*delta*100 :.1f}%")
    else:
        print("наличие Embarked не влияет на точность модели")

if __name__ == "__main__":
    main()