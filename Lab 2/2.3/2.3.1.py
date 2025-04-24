import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, precision_recall_curve,
                             roc_curve, roc_auc_score, average_precision_score)


def load_data(filepath):
    data = pd.read_csv(filepath)
    print("Данные загружены.")

    initial_num = len(data)

    for i in ['PassengerId', 'Name', 'Ticket', 'Cabin']:
        data = data.drop(i, axis=1)

    data = data.dropna()

    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'C': 1, 'Q': 2, 'S': 3})

    print(f"Утерянный процент данных: {100 * (initial_num - len(data)) / initial_num : .1f}%")

    return data

def sklearn_log_regression(data, label, target):
    y = data[target]
    x = data.drop(target, axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

    clf = LogisticRegression(random_state=0, max_iter=1000)
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    y_scores = clf.predict_proba(x_test)[:, 1]  # Вероятности для положительного класса

    # Вычисление метрик
    accuracy = clf.score(x_test, y_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nМодель sklearn, {label}")
    print(f"Точность (Accuracy): {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"AUC-ROC: {roc_auc_score(y_test, y_scores):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, y_scores):.4f}")

    # Визуализации
    plot_confusion_matrix(y_test, y_pred, f'Матрица ошибок ({label})')
    plot_pr_curve(y_test, y_scores, f'PR-кривая ({label})')
    plot_roc_curve(y_test, y_scores, f'ROC-кривая ({label})')

    return accuracy

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    plt.title(title)
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.show()

def plot_pr_curve(y_true, y_scores, title):
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR curve (AP = {average_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def plot_roc_curve(y_true, y_scores, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def main():
    data = load_data("Titanic.csv")

    data_without_emb = data.drop('Embarked', axis=1)

    print("\n" + "=" * 50)
    print("Модель с признаком Embarked")
    print("=" * 50)
    score_with_emb = sklearn_log_regression(data=data, label='with Embarked', target='Survived')

    print("\n" + "=" * 50)
    print("Модель без признака Embarked")
    print("=" * 50)
    score_without_emb = sklearn_log_regression(data=data_without_emb, label='without Embarked', target='Survived')

    delta = score_with_emb - score_without_emb
    if delta > 0:
        print(f"\nEmbarked улучшает точность модели на {delta * 100:.1f}%")
    elif delta < 0:
        print(f"\nEmbarked ухудшает точность модели на {-1 * delta * 100:.1f}%")
    else:
        print("\nНаличие Embarked не влияет на точность модели")

if __name__ == "__main__":
    main()

# Вывод: Модель показывает умеренное качество (Accuracy 78.5%, AUC-ROC 0.86), но страдает от дисбаланса классов (Precision и Recall ~73-74%).
# Признак Embarked практически не влияет на результат, поэтому его можно исключить без потерь.