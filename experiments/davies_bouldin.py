
import numpy as np


def _big_s(x, center):
    len_x, _ = x.shape
    total = 0

    if len_x is 0:
        return 0

    for i in range(len_x):
        total += np.linalg.norm(x[i]-center)

    return total / len_x


def davies_bouldin_score(X, labels_pred, k_centers):
    num_clusters, _ = k_centers.shape
    big_ss = np.zeros([num_clusters], dtype=np.float64)
    d_eucs = np.zeros([num_clusters, num_clusters], dtype=np.float64)
    db = 0

    for k in range(num_clusters):
        samples_in_k_inds = np.where(labels_pred == k)[0]
        samples_in_k = X[samples_in_k_inds, :]
        big_ss[k] = _big_s(samples_in_k, k_centers[k])

    for k in range(num_clusters):
        for l in range(0, num_clusters):
            d_eucs[k, l] = np.linalg.norm(k_centers[k]-k_centers[l])

    for k in range(num_clusters):
        values = np.zeros([num_clusters-1], dtype=np.float64)
        for l in range(0, k):
            values[l] = (big_ss[k] + big_ss[l]) / d_eucs[k, l]
        for l in range(k+1, num_clusters):
            values[l-1] = (big_ss[k] + big_ss[l]) / d_eucs[k, l]

        db += np.max(values)
    res = db / num_clusters
    return res


def calculate_centroids_doc_mean(X, labels_pred, k):
    _, m = X.shape

    centroids = np.zeros((k, m))
    for k in range(k):
        samples_in_k_inds = np.where(labels_pred == k)[0]
        centroids[k, :] = X[samples_in_k_inds, :].mean(axis=0)

    return centroids
