import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import sys
import scipy


def calc_euclidean_distance(a, b):
    return np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))


def calc_hamming_distance(a, b):
    return np.sum(np.absolute(np.subtract(a, b)))


def calc_manhattan_distance(a, b):
    return np.max(np.absolute(np.subtract(a, b)))


def calc_jaccard_distance(a, b):
    dist = (np.double(np.bitwise_and((a != b), np.bitwise_or(a != 0, b != 0)).sum())
            / np.double(np.bitwise_or(a != 0, b != 0).sum()))

    return dist


def calc_cos_distance(a, b):
    numerator = np.sum(np.multiply(a, b))
    denumerator = np.sqrt(np.sum(np.square(a))) * np.sqrt(np.sum(np.square(b)))
    dist = 1 - (numerator / denumerator)
    return dist


#  k-Nearest Neighbors
def kkn(x_test, y_test, x_train, y_train, n_neighbors: int, calc_dist):

    # Ищем расстоание от тестовых до всех тренировочных.
    dist = []
    for i in range(0, len(x_test)):
        dist_from_test = []
        for j in range(0, len(x_train)):
            dist_from_test.append((j, calc_dist(x_test[i], x_train[j]), y_train[j][0]))
        dist.append(dist_from_test)

    result = []  # Наиболее часто встречающиеся вершины среди n_neighbors ближайших для каждой тестовой.
    for p in dist:
        point_classes = {}
        p.sort(key=lambda x: x[1])
        neighbors = p[:n_neighbors]  # n ближайших соседей тестовой точки

        for neighbor in neighbors:
            n_name = neighbor[2]
            if n_name in point_classes:
                point_classes[n_name] += 1
            else:
                point_classes[n_name] = 1
        max_key = max(point_classes, key=lambda k: point_classes[k])
        result.append(max_key)

    # Вывод результатов с соотношением.
    src_set = {"setosa": 0, "versicolor": 0, "virginica": 0}
    correct_ident = {"setosa": 0, "versicolor": 0, "virginica": 0}
    for i in range(len(x_test)):
        if y_test[i][0] == result[i]:
            correct_ident[result[i]] += 1
        src_set[result[i]] += 1


    # print(src_set)
    # print(correct_ident)
    print("setosa", 0 if src_set["setosa"] == 0 else correct_ident["setosa"] / src_set["setosa"])
    print("versicolor", 0 if src_set["versicolor"] == 0 else correct_ident["versicolor"] / src_set["versicolor"])
    print("virginica", 0 if src_set["virginica"] == 0 else correct_ident["virginica"] / src_set["virginica"])
    print()


def main():
    dataset_path = "C:\dev\psu\pattern-recognition\PatternRecognition\iris.csv"
    dataset = pd.read_csv(dataset_path)
    coord = 4
    x = dataset.values[:, :coord]
    y = dataset.values[:, coord:]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    # Вывод результатов
    neighbors_count = 5
    kkn(x_test, y_test, x_train, y_train, neighbors_count, calc_euclidean_distance)

    # Сверка knn с помощью sklearn
    classifier = KNeighborsClassifier(n_neighbors=neighbors_count)
    classifier.fit(x_train, y_train)
    y_pred = classifier.predict(x_test)
    print(classification_report(y_test, y_pred))


