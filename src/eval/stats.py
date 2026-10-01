"""Small, dependency-light statistical helpers for clustered evaluation.

The independent unit in the public-data experiments is an animal (or source
file for the sheep proxy data), not an individual frame.  These helpers keep
that unit explicit and are deliberately usable from scripts and tests without
loading a model or a dataset.
"""

import numpy as np
from scipy.stats import beta
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             roc_auc_score)


DEFAULT_SEED = 42


def classification_summary(y_true, y_pred, y_train=None):
    """Report identity metrics and explicit constant-class baselines.

    Macro metrics average over classes present in the probe set. Uniform
    guessing draws from the training vocabulary when supplied (which can
    contain classes absent from the probe set). ``majority_accuracy`` is the
    largest *probe* class fraction, a descriptive oracle benchmark, not a
    train-selected classifier. Its deployable counterpart is reported below.
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if (y_true.ndim != 1 or y_pred.ndim != 1 or
            len(y_true) == 0 or len(y_true) != len(y_pred)):
        raise ValueError("true and predicted labels must be nonempty aligned vectors")
    labels, counts = np.unique(y_true, return_counts=True)
    k = len(labels)
    majority = float(counts.max() / len(y_true))
    train = y_true if y_train is None else np.asarray(y_train)
    if train.ndim != 1 or len(train) == 0:
        raise ValueError("training labels must be a nonempty vector")
    train_labels, train_counts = np.unique(train, return_counts=True)
    if not np.isin(labels, train_labels).all():
        raise ValueError("probe classes must be present in the training vocabulary")
    majority_class = train_labels[np.argmax(train_counts)]
    result = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "balanced_accuracy": round(float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0)), 4),
        "majority_accuracy": round(majority, 4),
        "uniform_chance": round(float(1 / len(train_labels)), 4),
        "balanced_constant_baseline": round(float(1 / k), 4),
        "n_classes": int(k),
        "n_train_classes": int(len(train_labels)),
        "n": int(len(y_true)),
        "macro_average_over": "probe-present classes",
        "majority_accuracy_definition": "largest probe class fraction (descriptive)",
    }
    if y_train is not None:
        result['train_majority_class'] = str(majority_class)
        result['train_majority_accuracy'] = round(float(np.mean(y_true == majority_class)), 4)
        result['train_majority_balanced_accuracy'] = (
            round(1 / k, 4) if majority_class in labels else 0.0)
    return result


def _validate_bootstrap(n_resamples, alpha):
    if not isinstance(n_resamples, (int, np.integer)) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between zero and one")


def cluster_bootstrap_ci(values, clusters, statistic=np.mean, n_resamples=1000,
                         seed=DEFAULT_SEED, alpha=0.05):
    """Percentile CI for an observation statistic resampling whole clusters.

    ``values`` and ``clusters`` are one-dimensional and aligned.  A cluster is
    sampled with replacement and all of its observations are retained.  This
    is suitable for frame-level scores where animal/source-file is the unit of
    independence.  The statistic is called on the concatenated resample.
    """
    _validate_bootstrap(n_resamples, alpha)
    values = np.asarray(values)
    clusters = np.asarray(clusters)
    if values.ndim != 1 or clusters.ndim != 1 or len(values) != len(clusters):
        raise ValueError("values and clusters must be aligned one-dimensional arrays")
    unique = np.unique(clusters)
    if len(unique) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    by_cluster = [np.flatnonzero(clusters == c) for c in unique]
    estimates = []
    for _ in range(int(n_resamples)):
        chosen = rng.randint(0, len(by_cluster), len(by_cluster))
        ix = np.concatenate([by_cluster[i] for i in chosen])
        value = statistic(values[ix])
        if np.isfinite(value):
            estimates.append(float(value))
    if not estimates:
        return [None, None]
    return [round(float(np.percentile(estimates, 100 * alpha / 2)), 4),
            round(float(np.percentile(estimates, 100 * (1 - alpha / 2))), 4)]


def cluster_bootstrap_auc(y_true, scores, clusters, n_resamples=1000,
                          seed=DEFAULT_SEED, alpha=0.05):
    """Cluster bootstrap CI for AUROC, returning ``[low, high]``."""
    _validate_bootstrap(n_resamples, alpha)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    clusters = np.asarray(clusters)
    if not (y_true.ndim == scores.ndim == clusters.ndim == 1 and
            len(y_true) == len(scores) == len(clusters)):
        raise ValueError("y_true, scores and clusters must be aligned")
    unique = np.unique(clusters)
    if len(unique) == 0:
        return [None, None]
    rng = np.random.RandomState(seed)
    by_cluster = [np.flatnonzero(clusters == c) for c in unique]
    estimates = []
    for _ in range(int(n_resamples)):
        chosen = rng.randint(0, len(by_cluster), len(by_cluster))
        ix = np.concatenate([by_cluster[i] for i in chosen])
        if len(np.unique(y_true[ix])) < 2:
            continue
        estimates.append(float(roc_auc_score(y_true[ix], scores[ix])))
    if not estimates:
        return [None, None]
    return [round(float(np.percentile(estimates, 100 * alpha / 2)), 4),
            round(float(np.percentile(estimates, 100 * (1 - alpha / 2))), 4)]


def paired_cluster_mean_ci(a, b, clusters, n_resamples=1000,
                           seed=DEFAULT_SEED, alpha=0.05):
    """CI for the equal-cluster-weighted paired mean difference ``a - b``.

    Average within each cluster first, so cows with more frames/directions
    do not receive more weight than cows with fewer observations.
    """
    _validate_bootstrap(n_resamples, alpha)
    a, b, clusters = np.asarray(a), np.asarray(b), np.asarray(clusters)
    if not (a.ndim == b.ndim == clusters.ndim == 1 and len(a) == len(b) == len(clusters)):
        raise ValueError("a, b and clusters must be aligned one-dimensional arrays")
    unique = np.unique(clusters)
    if len(unique) == 0:
        return [None, None]
    diff = a - b
    cluster_means = np.array([diff[clusters == c].mean() for c in unique])
    if not np.isfinite(cluster_means).all():
        raise ValueError("paired differences must be finite")
    rng = np.random.RandomState(seed)
    estimates = []
    for _ in range(n_resamples):
        chosen = rng.randint(0, len(unique), len(unique))
        estimates.append(float(np.mean(cluster_means[chosen])))
    return [round(float(np.percentile(estimates, 100 * alpha / 2)), 4),
            round(float(np.percentile(estimates, 100 * (1 - alpha / 2))), 4)]


def binomial_miss_upper_bound(misses, n_events, alpha=0.05):
    """One-sided exact upper bound on a miss probability.

    With zero misses this is ``1 - alpha**(1/n_events)``; reporting it prevents
    a small event count from being presented as proof of zero miss risk.
    """
    if (not isinstance(misses, (int, np.integer)) or
            not isinstance(n_events, (int, np.integer)) or
            n_events <= 0 or misses < 0 or misses > n_events or not 0 < alpha < 1):
        raise ValueError("invalid event count or alpha")
    if misses == n_events:
        return 1.0
    return float(beta.ppf(1 - alpha, misses + 1, n_events - misses))
