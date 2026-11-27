"""Reproducible, exclusive outer folds and spatially isolated inner folds."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, KFold

METHODS = ("GroupKFold", "BlockKFold", "BufferedKFold", "KFold", "ShuffleKFold")
SPATIAL_METHODS = {"BlockKFold", "BufferedKFold"}


def group_labels(method, xy, labels=None, block_size=10000., origin=None):
    if method == "GroupKFold":
        if labels is None or len(labels) != len(xy) or pd.isna(labels).any():
            raise ValueError("GroupKFold: выберите столбец групп без пропусков.")
        return pd.factorize(labels, sort=False)[0]
    if method == "BlockKFold":
        if not np.isfinite(block_size) or block_size <= 0:
            raise ValueError("Размер блока должен быть > 0.")
        blocks = np.floor((xy - (xy.min(axis=0) if origin is None else origin)) / block_size)
        if not np.isfinite(blocks).all():
            raise ValueError("Некорректная сетка пространственных блоков.")
        return np.unique(blocks, axis=0, return_inverse=True)[1]
    return None


def make_splits(method, xy, n_splits, *, labels=None, block_size=10000.,
                buffer_size=1000., seed=42, origin=None, inner=False):
    """Each sample is tested once; BufferedKFold uses plain KFold then a distance filter.

    BufferedKFold does not use square blocks, grouping, or block_size.
    The order of input rows determines its sequential control folds.
    """
    xy = np.asarray(xy, dtype=float)
    if method not in METHODS:
        raise ValueError("Неизвестный метод валидации.")
    if not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("Количество фолдов должно быть не менее 2.")
    if xy.ndim != 2 or xy.shape[1] != 2 or not np.isfinite(xy).all():
        raise ValueError("Необходимы конечные двумерные координаты.")
    if method == "BufferedKFold" and (not np.isfinite(buffer_size) or buffer_size < 0):
        raise ValueError("Буфер должен быть неотрицательным.")
    groups = group_labels(method, xy, labels, block_size, origin)
    maximum = len(xy) // 2
    if groups is not None:
        maximum = min(maximum, len(np.unique(groups)))
    actual = min(n_splits, maximum) if inner else n_splits
    if actual < 2 or actual > maximum:
        raise ValueError(f"{method}: недостаточно групп/точек для {n_splits} фолдов.")
    if groups is None:
        shuffle = method == "ShuffleKFold"
        raw = KFold(actual, shuffle=shuffle, random_state=seed if shuffle else None).split(xy)
    else:
        raw = GroupKFold(actual).split(xy, groups=groups)
    splits = []
    for train, valid in raw:
        if method == "BufferedKFold":
            distance, _ = cKDTree(xy[valid]).query(xy[train], k=1)
            train = train[distance > buffer_size]
        if min(len(train), len(valid)) < 2:
            raise ValueError(f"{method}: фолд {len(splits)+1} содержит менее двух точек "
                             "обучения/контроля. Измените число фолдов или параметры выбранной схемы.")
        splits.append((train, valid))
    if not np.array_equal(np.sort(np.concatenate([v for _, v in splits])), np.arange(len(xy))):
        raise RuntimeError("Нарушено покрытие контрольных фолдов.")
    return splits
