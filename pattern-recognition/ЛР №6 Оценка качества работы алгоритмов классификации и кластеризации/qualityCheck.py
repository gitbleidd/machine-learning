import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

import maximin

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import r2_score
from sklearn.metrics import silhouette_score

def main():
    dataset_path = "C:\dev\psu\pattern-recognition\PatternRecognition\iris.csv"
    dataset = pd.read_csv(dataset_path)


    coord = 4
    X = dataset.values[:, :coord]
    Y = dataset.values[:, coord:]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1)

    p_kmeans, kmeans_cluster_centers = kmeans_predict(x_train, x_test, y_train, y_test)
    p_knn = knn_predict(x_train, x_test, y_train, y_test)
    p_svm = svm_predict(x_train, x_test, y_train, y_test)
    p_maximin = maximin_predict(x_train, x_test, y_train, y_test)

    # print(p_knn)
    # print(p_svm)

    print("KNN")
    y_pred = p_knn
    y_test_nums, y_pred_nums = label_to_nums(dataset, y_test, y_pred)
    test_prediction(x_test, np.array(y_test_nums), np.array(y_pred_nums), "micro")
    print("\n\n\n")

    print("SVM")
    y_pred = p_svm
    y_test_nums, y_pred_nums = label_to_nums(dataset, y_test, y_pred)
    test_prediction(x_test, np.array(y_test_nums), np.array(y_pred_nums), "micro")
    print("\n\n\n")

    # Clustering to classification
    labels_relation = clustering_to_classification(x_test, y_test, p_kmeans)  # Показывает отношение predicted -> y_test labels
    labels_names = np.unique(y_test)
    p_kmeans_with_labels = []
    for i in range(p_kmeans.size):
        label_num = labels_relation[p_kmeans[i]]
        p_kmeans_with_labels.append(labels_names[label_num])
    p_kmeans = np.array(p_kmeans_with_labels)

    print("KMeans")
    y_pred = p_kmeans
    y_test_nums, y_pred_nums = label_to_nums(dataset, y_test, y_pred)
    test_prediction(x_test, np.array(y_test_nums), np.array(y_pred_nums), "micro")


    # p_maximin = np.array(p_maximin)
    # print(p_maximin)
    # print(clustering_to_classification(x_test, y_test, p_maximin))


def clustering_to_classification(x_test, y_test, y_pred):
    pred_clust = y_pred
    labels_count = np.unique(y_test).size
    clusters_count = np.unique(pred_clust).size
    test_centers = calc_centers(x_test, y_test)

    labes_with_pred = [[] for i in range(labels_count)]
    for i in range(pred_clust.size):
        closest_label = -1
        min_dist = float("inf")
        for j in range(labels_count):
            dist = np.linalg.norm(test_centers[j] - x_test[i])
            if dist < min_dist:
                min_dist = dist
                closest_label = j
        labes_with_pred[closest_label].append(pred_clust[i])

    # res = []
    # for i in labes_with_pred:
    #     res.append(max(set(i), key=i.count))
    # return(res)
    res = {}
    for i in np.unique(y_pred):
        res[i] = -1

    for i in range(len(labes_with_pred)):
        pred_element = max(set(labes_with_pred[i]), key=labes_with_pred[i].count)
        res[pred_element] = i
    return res

def label_to_nums(dataset, y_test: np.array, y_pred: np.array):
    names = dataset["species"].to_numpy()
    names = pd.unique(names)
    y_test_nums = []
    y_pred_nums = []
    names_nums = {}
    for i in range(len(names)):
        names_nums[names[i]] = i
    for i in range(len(y_test)):
        i1 = y_test[i].item()
        i2 = y_pred[i]
        y_test_nums.append(names_nums[i1])
        y_pred_nums.append(names_nums[i2])

    return y_test_nums, y_pred_nums


def test_prediction(x_test, y_test, pred, recall_precision_mode="micro"):
    recall, precision = calc_recall_precision(calc_confusion_matrix(y_test, pred), recall_precision_mode)
    f1 = calc_f1(recall, precision)

    y_test_nums = y_test
    y_pred_nums = pred
    r2 = calc_determination_coefficient(np.array(y_test_nums), np.array(y_pred_nums))

    cluster_centers = calc_centers(x_test, pred)
    silhouette_coef = calc_silhouette_coefficient(x_test, pred, cluster_centers)
    dunn_index = calc_dunn_index(x_test, pred, cluster_centers)

    print("Mode:", recall_precision_mode)
    print("recall: ", recall)
    print("precision: ", precision)
    print("f1: ", f1)
    print("R2: ", r2)
    print("Silhouette coefficient: ", silhouette_coef)
    print("Dunn index: ", dunn_index)

    print("--- --- --- ---")
    print("Проверка с sklearn.metrics:")
    print("accurancy: ", accuracy_score(y_test, pred))
    print("recall: ", recall_score(y_test, pred, average=recall_precision_mode))
    print("precision: ", precision_score(y_test, pred, average=recall_precision_mode))
    print("R2: ", r2_score(y_test_nums, y_pred_nums))
    print("Silhouette coefficient", silhouette_score(x_test, pred))


def kmeans_predict(x_train, x_test, y_train, y_test):
    kmeans = KMeans(n_clusters=3)  #TODO
    kmeans.fit(x_train)
    predictions = kmeans.predict(x_test)

    return predictions, kmeans.cluster_centers_


def knn_predict(x_train, x_test, y_train, y_test):
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(x_train, y_train)
    predictions = neigh.predict(x_test)

    return predictions


def svm_predict(x_train, x_test, y_train, y_test):
    alpha = 1.0
    svm_model = svm.SVC(kernel='linear', C=alpha)

    svm_model.fit(x_train, y_train)
    predictions = svm_model.predict(x_test)

    return predictions


def maximin_predict(x_train, x_test, y_train, y_test):
    maximin.fit_with_split(x_train, y_train)
    predictions = maximin.predict(x_test, x_train)

    return predictions

# Метрики

def calc_confusion_matrix(Y:np.array, prediction):
    labels = np.unique(Y)
    label_num = {}
    for i, label in enumerate(labels):
        label_num[label] = i

    n = len(labels)
    matrix = [[0 for i in range(n)] for i in range(n)]

    for i, y in enumerate(Y):
        row = label_num[y.item()]
        col = label_num[prediction[i]]
        matrix[col][row] += 1

    # print(Y)
    # print(prediction)
    print("Confusion matrix:")
    for r in matrix:
        print(r)
    print("")

    return matrix


# Precision можно интерпретировать как долю объектов,
# названных классификатором положительными и при этом действительно являющимися положительными
#
# Recall показывает, какую долю объектов положительного класса
# из всех объектов положительного класса нашел алгоритм.
def calc_recall_precision(confusion_matrix, level: str):
    arr = np.array(confusion_matrix)

    rows = np.sum(arr, axis=1)  # (TP(i) + FN(i))
    columns = np.sum(arr, axis=0)  # (TP(i) + FP(i))
    diagonals = np.diag(arr)

    if level == "micro":
        # TP(i) / (TP(i) + FN(i))
        recall = sum(diagonals) / sum(rows)

        # sum of TP(i) / sum of (TP(i) + FP(i))
        precision = sum(diagonals) / sum(columns)

    elif level == "macro":
        # sum recall(i) / classes_count
        recall = sum((diagonals / rows)) / len(diagonals)

        # sum precision(i) / classes_count
        precision = sum((diagonals / columns)) / len(diagonals)

    return recall, precision


# Accuracy: Хорошо работает на сбалансированных данных = TP / кол-во всех
# F1 score: применятся в тех случаях, когда данные несбалансированые
def calc_f1(recall, precision):
    f = 2 * precision * recall / (precision + recall)
    return f


# MSE - Среднеквадратичная ошибка
def calc_mean_squared_error(y_true, y_pred):
    return np.square(y_true - y_pred).mean()


# Коэффициент детерминации
def calc_determination_coefficient(y_true, y_pred):
    num = np.sum(np.square(y_pred - y_true))
    y_avg = np.average(y_true)
    denum = np.sum((np.subtract(y_true, y_avg))**2)

    numerator = ((y_true - y_pred) ** 2).sum()
    denominator = ((y_true - np.average(y_true)) ** 2).sum()

    return 1 - (num / denum)


# Если значение силуэта близко к 1, элемент хорошо кластеризован.
# Если значение силуэта близко к 0, элемент может быть отнесен к другому ближайшему к нему кластеру.
# Если значение силуэта близко к –1, образец неправильно классифицируется.
def calc_silhouette_coefficient(x_test, y_pred: np.ndarray, cluster_centers: np.ndarray):
    # clusters_nums = np.unique(y_pred)
    # clusters = list(zip(x_test, y_pred))
    #
    # # Поиск центров
    # if len(cluster_centers) == 0:
    #     for c in clusters_nums:
    #         count = 0
    #         center = np.zeros(len(x_test[0]))
    #         for i in range(len(y_pred)):
    #             if clusters[i][1] == c:
    #                 for el_i in range(len(clusters[i][0])):
    #                     center[el_i] += clusters[i][0][el_i]
    #                 count += 1
    #         center /= count
    #         cluster_centers.append(center)

    a = []
    b = []
    s = []
    for i in range(len(x_test)):
        a_cur = 0
        a_count = 0

        cluster_num = y_pred[i]
        for j in range(len(x_test)):
            # Если не из этого кластера или этот же элемент
            if i == j or y_pred[j] != cluster_num:
                continue

            a_cur += np.linalg.norm(x_test[i] - x_test[j])
            a_count += 1

        if a_count == 0:
            a_count = 1
        a.append(a_cur/a_count)

        # Ищем ближайший кластер к кластеру i-го элемента
        closest_cluster = -1
        min_dist = float("inf")
        for c in range(len(cluster_centers)):
            if c == cluster_num.item():
                continue
            d = np.linalg.norm(cluster_centers[cluster_num] - cluster_centers[c])
            if d < min_dist:
                min_dist = d
                closest_cluster = c

        b_cur = 0
        b_count = 0
        for j in range(len(x_test)):
            if y_pred[j] != closest_cluster:
                continue

            b_cur += np.linalg.norm(x_test[i] - x_test[j])
            b_count += 1

        if b_count == 0:
            b_count = 1
        b.append(b_cur/b_count)

    for i in range(len(a)):
        if a[i] == 0 or b[i] == 0:
            s.append(0.0)
            continue
        s_cur = (b[i] - a[i]) / max(a[i], b[i])
        s.append(s_cur)
    return np.average(s)


# Чем выше значение индекса Данна, тем лучше кластеризация.
# Недостаток - вычислительные затраты
def calc_dunn_index(x_test, y_pred: np.ndarray, cluster_centers: np.ndarray):
    # Расстояние между кластерами
    numerator = float('inf')
    for i, c1 in enumerate(cluster_centers):
        for j, c2 in enumerate(cluster_centers):
            if i == j:
                continue
            cur_dist = np.linalg.norm(c1 - c2)
            numerator = min(numerator, cur_dist)

    # Расстояние между точками внутри кластера
    denumerator = 0
    for i, c in enumerate(cluster_centers):
        for p1 in range(len(x_test)):
            for p2 in range(len(x_test)):
                if p1 == p2 or y_pred[p1] != y_pred[p2]:
                    continue
                cur_dist = np.linalg.norm(x_test[p1] - x_test[p2])
                denumerator = max(denumerator, cur_dist)

    return numerator / denumerator


def calc_centers(x_test, y_pred):
    cluster_centers = []
    clusters_nums = np.unique(y_pred)
    clusters = list(zip(x_test, y_pred))

    for c in clusters_nums:
        count = 0
        center = np.zeros(len(x_test[0]))
        for i in range(len(y_pred)):
            if clusters[i][1] == c:
                for el_i in range(len(clusters[i][0])):
                    center[el_i] += clusters[i][0][el_i]
                count += 1
        center /= count
        cluster_centers.append(center)

    return np.array(cluster_centers)
