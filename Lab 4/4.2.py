import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_data():
    digits = load_digits()

    print("\n=== Информация о датасете ===")
    print(f"Количество изображений: {len(digits.images)}")
    print(f"Размер изображений: {digits.images[0].shape}")
    print(f"Количество классов: {len(np.unique(digits.target))}")

    plt.figure(figsize=(10, 4))
    for index, (image, label) in enumerate(zip(digits.images[:10], digits.target[:10])):
        plt.subplot(2, 5, index + 1)
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"Label: {label}")
    plt.tight_layout()
    plt.show()

    return digits


def preprocess_data(digits):
    n_samples = len(digits.images)
    X = digits.images.reshape((n_samples, -1))
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    print("\n=== Обучение модели MLPClassifier ===")

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500,
                        alpha=1e-4, solver='adam',
                        verbose=10, random_state=1,
                        learning_rate_init=0.001)

    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)

    print("\n=== Оценка модели ===")
    print(f"Общая точность: {accuracy_score(y_test, y_pred):.4f}")
    print("\nОтчет по классификации:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Матрица ошибок")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Истинные метки')
    plt.xlabel('Предсказанные метки')
    plt.tight_layout()
    plt.show()

    return mlp


def visualize_incorrect_predictions(X_test, y_test, y_pred, images):
    incorrect = np.where(y_test != y_pred)[0]

    print(f"\nКоличество неправильных предсказаний: {len(incorrect)}")

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(incorrect[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(images[idx].reshape(8, 8), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title(f"True: {y_test[idx]}\nPred: {y_pred[idx]}")
    plt.tight_layout()
    plt.show()


def main():
    digits = load_data()

    X_train, X_test, y_train, y_test = preprocess_data(digits)

    model = train_and_evaluate_model(X_train, X_test, y_train, y_test)

    X_test_original = digits.images[len(digits.images) - len(X_test):]
    y_pred = model.predict(X_test)
    visualize_incorrect_predictions(X_test, y_test, y_pred, X_test_original)

    print("\n=== Выводы ===")
    print("1. MLPClassifier показывает хорошую точность (обычно >95%) на датасете рукописных цифр\n"
    "2. Масштабирование данных критически важно для работы нейронной сети\n"
    "3. Наиболее сложные для классификации цифры часто: 8, 3, 5, 9 (похожи по начертанию)\n"
    "4. Для улучшения результатов можно экспериментировать с:\n"
    "   - Архитектурой сети (количество слоев и нейронов)\n"
    "   - Параметрами обучения (скорость обучения, регуляризация)\n"
    "   - Увеличением количества эпох обучения")


if __name__ == "__main__":
    main()