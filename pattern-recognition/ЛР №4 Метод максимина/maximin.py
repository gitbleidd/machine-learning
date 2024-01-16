import numpy as np
import pandas as pd
import random
from itertools import combinations


def calc_euclidean_distance(a, b):
    n = len(a)
    sum = 0
    for i in range(0, n - 1):  # n-1, т.к. входной вектор содержит категорию
        sum += pow((a[i] - b[i]), 2)
    return np.sqrt(sum)


def main():
    dataset_path = "C:\dev\psu\pattern-recognition\iris.csv"
    dataset = pd.read_csv(dataset_path).values
    size = len(dataset)

    patterns = set()
    for i in range(0, size):
        patterns.add(i)

    centroids = []
    centroid = random.randint(0, size - 1)  # Первый центр берем произвольный.
    centroids.append(centroid)
    patterns.remove(centroid)

    # Ищем второй центр (на макс расст. от 1-го)
    centroid2_dist = -1
    centroid2_idx = -1
    for i in range(0, size):
        cur_dist = calc_euclidean_distance(dataset[centroid], dataset[i])
        if cur_dist > centroid2_dist:
            centroid2_dist = cur_dist
            centroid2_idx = i

    centroids.append(centroid2_idx)
    patterns.remove(centroid2_idx)

    is_new_cluster = True
    while is_new_cluster:
        # Находим максимальное расстояние из минимальных
        max_min_dist = -1
        max_min_idx = -1
        for p in patterns:
            min_dist = float('inf')
            min_dist_idx = -1
            for c in centroids:
                cur_dist = calc_euclidean_distance(dataset[p], dataset[c])
                if cur_dist < min_dist:
                    min_dist_idx = p
                    min_dist = cur_dist

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                max_min_idx = min_dist_idx

        mean_distance = np.average(list(calc_euclidean_distance(dataset[x], dataset[y]) for x, y in combinations(centroids, 2)))
        if (max_min_dist > mean_distance / 2.0):
            centroids.append(max_min_idx)
            patterns.remove(max_min_idx)
        else:
            is_new_cluster = False
    print(centroids)

    res = {}
    for c in centroids:
        res[c] = []
    for p in patterns:
        dists = [(x, calc_euclidean_distance(dataset[p], dataset[x])) for x in list(res.keys())]
        min_dist = min(dists, key=lambda x: x[1])
        res[min_dist[0]].append(p)

    for i in res.keys():
        print('Centroid:', i, res[i])







