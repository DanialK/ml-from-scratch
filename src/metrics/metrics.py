import numpy as np


def accuracy(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    return np.mean(y == y_pred)


def confusion_matrix(y, y_pred):
    y, y_pred = np.array(y), np.array(y_pred)
    labels = np.sort(np.unique(y))
    n_labels = len(labels)
    co_matrix = np.zeros((n_labels, n_labels), dtype=int)
    for i, label in enumerate(labels):
        idx = y == label
        counts = np.bincount(y_pred[idx])
        co_matrix[i, range(counts.shape[0])] = np.array(counts)
    return co_matrix


def precision(y, y_pred, co_matrix=None):
    y, y_pred = np.array(y), np.array(y_pred)
    if co_matrix is None:
        co_matrix = confusion_matrix(y, y_pred)
    tp = co_matrix[1][1]
    fp = co_matrix[0][1]
    return tp / (tp + fp)


def recall(y, y_pred, co_matrix=None):
    y, y_pred = np.array(y), np.array(y_pred)
    if co_matrix is None:
        co_matrix = confusion_matrix(y, y_pred)
    tp = co_matrix[1][1]
    fn = co_matrix[1][0]
    return tp / (tp + fn)


sensitivity = recall

true_positive_rate = recall


def specificity(y, y_pred, co_matrix=None):
    y, y_pred = np.array(y), np.array(y_pred)
    if co_matrix is None:
        co_matrix = confusion_matrix(y, y_pred)
    tn = co_matrix[0][0]
    fp = co_matrix[0][1]
    return tn / (tn + fp)


true_negative_rate = specificity


def false_positive_rate(y, y_pred, co_matrix=None):
    y, y_pred = np.array(y), np.array(y_pred)
    return 1 - true_negative_rate(y, y_pred, co_matrix)


def f1_score(y, y_pred, co_matrix=None):
    y, y_pred = np.array(y), np.array(y_pred)
    if co_matrix is None:
        co_matrix = confusion_matrix(y, y_pred)
    p = precision(y, y_pred, co_matrix)
    r = recall(y, y_pred, co_matrix)
    return 2 * (p * r) / (p + r)


def roc(y, y_prob, thresholds=None):
    y, y_prob = np.array(y), np.array(y_prob)

    if thresholds is None:
        thresholds = list(np.unique(y_prob)) + [1 + np.max(y)]
    else:
        thresholds = np.sort(thresholds)

    points = np.zeros((3, len(thresholds)))
    points[0] = thresholds

    for i, threshold in enumerate(thresholds):
        y_pred = y_prob >= threshold
        co_matrix = confusion_matrix(y, y_pred)
        tpr = true_positive_rate(y, y_pred, co_matrix)
        fpr = false_positive_rate(y, y_pred, co_matrix)
        points[1][i] = fpr
        points[2][i] = tpr

    not_zero_idx = np.invert((points[1] == 0) & (points[2] == 0))

    return points[:, not_zero_idx]


def auc(y, y_prob, thresholds=None):
    y, y_prob = np.array(y), np.array(y_prob)
    roc_values = roc(y, y_prob, thresholds)
    fpr = roc_values[1][::-1]
    trp = roc_values[2][::-1]
    points1 = zip(fpr[:-1], trp[:-1])
    points2 = zip(fpr[1:], trp[1:])
    auc_val = 0
    for point1, point2 in zip(points1, points2):
        x = point2[0] - point1[0]
        y1 = point1[1]
        y2 = point2[1]
        area = (y1 + y2) * x / 2
        auc_val += area
    return auc_val


def log_loss(y, y_prob, epsilon = 1e-15):
    y, y_prob = np.array(y), np.array(y_prob)
    y_prob = np.clip(y_prob, epsilon, 1 - epsilon)
    return -1 * np.mean(y * np.log(y_prob) + (1-y) * np.log(1-y_prob))



