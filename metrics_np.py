"""
metrics_np.py — pure-numpy replacements for the two scikit/scipy calls the
validated harness used: roc_auc_score and scipy.stats.spearmanr. Keeping them
here (and tested against known values) lets the whole module stay numpy-only
without changing the science.
"""
from __future__ import annotations
import numpy as np


def _rankdata(x):
    """Average ranks (1-based), ties averaged — matches scipy.stats.rankdata('average')."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    n = len(x)
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0      # average of 1-based ranks i+1..j+1
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def roc_auc(labels, scores):
    """Binary AUROC via the Mann-Whitney U statistic on ranks (tie-aware).
    labels: 0/1 array; scores: higher => more positive. Returns float in [0,1]."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _rankdata(scores)
    sum_pos = ranks[labels == 1].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def spearman(a, b):
    """Spearman rank correlation = Pearson correlation of the ranks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom < 1e-12:
        return float("nan")
    return float((ra * rb).sum() / denom)
