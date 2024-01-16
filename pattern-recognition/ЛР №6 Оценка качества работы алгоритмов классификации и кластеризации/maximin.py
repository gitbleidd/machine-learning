import numpy as np
import pandas as pd
import random
from itertools import combinations

centroids = []


def euc_dist(a, b):
    return np.sqrt(np.sum(np.power(np.subtract(a, b), 2)))


def fit_with_split(x_train, y_train):
    size = x_train.shape[0]

    patterns = set()
    for i in range(0, size):
        patterns.add(i)

    centroid = random.randint(0, size - 1)  # Первый центр берем произвольный.
    centroids.append(centroid)
    patterns.remove(centroid)

    # Ищем второй центр (на макс расст. от 1-го)
    centroid2_dist = -1
    centroid2_idx = -1
    for i in range(0, size):
        cur_dist = euc_dist(x_train[centroid], x_train[i])
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
                cur_dist = euc_dist(x_train[p], x_train[c])

                if cur_dist < min_dist:
                    min_dist_idx = p
                    min_dist = cur_dist

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                max_min_idx = min_dist_idx

        mean_distance = np.average(list(euc_dist(x_train[x], x_train[y]) for x, y in combinations(centroids, 2)))
        if (max_min_dist > mean_distance / 2.0):
            centroids.append(max_min_idx)
            patterns.remove(max_min_idx)
        else:
            is_new_cluster = False
    # print(centroids)


def predict(x_test, x_train):
    centroid_nums = {}

    for c in range(len(centroids)):
        centroid_nums[c] = []

    res = []
    for p in x_test:
        dists = [(center, euc_dist(p, x_train[center])) for center in list(centroid_nums.keys())]
        min_dist = min(dists, key=lambda x: x[1])
        centroid_nums[min_dist[0]].append(p)
        res.append(min_dist[0])

    # for i in centroid_nums.keys():
    #     print('Centroid:', i, centroid_nums[i])
    #
    # print(centroid_nums.keys())
    # print(res)
    return res