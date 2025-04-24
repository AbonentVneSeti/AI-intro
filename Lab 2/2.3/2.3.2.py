import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, average_precision_score,
                             confusion_matrix, PrecisionRecallDisplay, RocCurveDisplay)
from sklearn.preprocessing import StandardScaler

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

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_proba),
        'PR AUC': average_precision_score(y_test, y_proba)
    }

    print(f"\n{model_name} метрики:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(model_name)

    # Матрица ошибок
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Survived', 'Survived'],
                yticklabels=['Not Survived', 'Survived'])
    ax1.set_title('Матрица ошибок')

    # ROC and PR
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax2, name='ROC')
    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax2, name='PR')
    ax2.set_title('кривые ROC & PR')

    plt.show()

    return metrics

def main():
    data = load_data('Titanic.csv')
    X = data.drop('Survived', axis=1)
    y = data['Survived']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=1),
        'SVM': SVC(probability=True, random_state=1),
        'KNN': KNeighborsClassifier()
    }

    results = {}
    for name, model in models.items():
        if name in ['SVM', 'KNN']:
            model.fit(X_train_scaled, y_train)
            metrics = evaluate_model(model, X_test_scaled, y_test, name)
        else:
            model.fit(X_train, y_train)
            metrics = evaluate_model(model, X_test, y_test, name)
        results[name] = metrics

    print("\nСравнение моделей:")
    comparison_df = pd.DataFrame(results).T
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(comparison_df)

    best_model = comparison_df['F1'].idxmax()
    print(f"\nЛучшая модель на основе метрики F1: {best_model}")

if __name__ == "__main__":
    main()