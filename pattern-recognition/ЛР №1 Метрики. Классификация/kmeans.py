import numpy as np
import pandas as pd
import random
from itertools import combinations


def calc_euclidean_distance(a, b):
    n = len(a)
    sum = 0
    for i in range(0, n):
        sum += pow((a[i] - b[i]), 2)
    return np.sqrt(sum)


def main():
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    dataset_path = "C:\dev\psu\pattern-recognition\iris.csv"
    dataset = pd.read_csv(dataset_path)
    x = dataset.values[:, :4]
    y = dataset.values[:, 4:]

    kmeans(x, y, 3)


def kmeans(x, y, k: int):
    size = len(x)

    # Исходные центры кластеров - первые k объектов
    centroids = x[:k]

    is_new_iter = True
    iter_count = 0
    while is_new_iter:

        # Находим какие точки принадлежат центрам кластеров.
        centroid_points = [[] for i in range(k)]
        for point in x:
            c_num = np.argmin([calc_euclidean_distance(point, centroid) for centroid in centroids])
            centroid_points[c_num].append(point)

        # Пересчитываем координаты центров.
        new_centers = []
        for i in range(0, k):
            # new_centers.append(np.mean([centroid_points[i][x] for x in range(len(centroid_points))], axis=0))
            new_centers.append(np.divide(np.sum(centroid_points[i], axis=0), float(len(centroid_points[i]))))
            # print("prev centroids:", centroids[i], "new:", new_centers[i])
            centroids[i] = new_centers[i]
        print()

        iter_count += 1
        if iter_count == 50:
            is_new_iter = False

    centroid_points = [[] for i in range(k)]
    for i in range(len(x)):
        point = x[i]
        c_num = np.argmin([calc_euclidean_distance(point, centroid) for centroid in centroids])
        centroid_points[c_num].append(i)

    for c in centroid_points:
        print(c)
        clusters_points = {"setosa":0, "versicolor":0, "virginica":0}
        for point in c:
            # print(point, y[point][0])
            clusters_points[y[point][0]] += 1
        print(clusters_points)
        print()








