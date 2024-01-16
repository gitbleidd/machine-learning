import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    dataset_path = "C:\dev\psu\pattern-recognition\PatternRecognition\iris.csv"
    dataset = pd.read_csv(dataset_path)
    coord = 4
    X = dataset.values[:, :coord]
    Y = dataset.values[:, coord:]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    # Сводим имена классов к числам: setosa, versicolor -> 1; virginica -> -1
    y_train_binary = convert_to_binary_labels(y_train)

    w_res = fit(x_train, y_train_binary, epoches=100, alpha=0.01, grad_step=0.01)
    prediction = predict(x_test, w=w_res)

    # Вывод результатов
    # print(y_test)
    # print(prediction)
    res = {-1: 0, 1: 0}
    y_test_binary = convert_to_binary_labels(y_test)
    for i, predicted in enumerate(prediction):
        if predicted == y_test_binary[i]:
            res[predicted] += 1

    first_class_count = sum(1 for i in y_test_binary if i == -1)
    second_class_count = len(y_test_binary) - first_class_count
    print("First class:", res[-1], "/", first_class_count)
    print("Second class:", res[1], "/", second_class_count)



def convert_to_binary_labels(Y:np.array) -> np.array:
    # return np.array([1 if item != "virginica" else -1 for item in Y])
    return np.array([1 if item != "versicolor" else -1 for item in Y])


def fit(X:np.array, Y:np.array, epoches, alpha, grad_step):
    # Расширяем вектор w на 1 и матрицу X на 1 столбец,
    # т.к. включаем сводный член b в формуле w^T * x - b.
    X_extended = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    w = np.random.normal(loc=0, scale=0.05, size=X_extended.shape[1])


    epoch_errors = []
    for epoch in range(epoches):
        epoch_errors.append(0)

        for i, x in enumerate(X_extended):
            margin = Y[i] * (w @ x)
            if margin >= 1:                 # Классифирован верно.
                dq_dw = alpha * w
            else:                           # Классифицирован неверно.
                dq_dw = alpha * w - Y[i] * x
                epoch_errors[epoch] += 1
            w = w - grad_step * dq_dw
    # print("Error on each epoch:")
    # print(epoch_errors)
    return w


def soft_margin_loss(y, x, w, alpha):
    return max(0, 1 - y * (x @ w)) + alpha * (w @ w) / 2.0


def predict(X, w):
    X_extended = np.append(X, np.ones((X.shape[0], 1)), axis=1)
    return [int(np.sign(w @ x)) for x in X_extended]
