"""
Compute handcrafted features from balanced group presentations.

Features capture structural properties of presentations that correlate
with distance to the trivial presentation under AC moves.
"""

import numpy as np


def compute_features(presentation, max_relator_length=18):
    """
    Compute 14 handcrafted features from a presentation.

    Parameters:
        presentation: int8 numpy array of length 2*max_relator_length.
                      First half is relator 1, second half is relator 2.
                      Values in {-2, -1, 0, 1, 2} with 0 as padding.
        max_relator_length: int, half the array length.

    Returns:
        np.ndarray of shape (14,), dtype float32.

    Features:
        0:  total_length (len_r1 + len_r2)
        1:  len_r1
        2:  len_r2
        3:  count of +1 (x) in relator 1
        4:  count of -1 (x^-1) in relator 1
        5:  count of +2 (y) in relator 1
        6:  count of -2 (y^-1) in relator 1
        7:  count of +1 (x) in relator 2
        8:  count of -1 (x^-1) in relator 2
        9:  count of +2 (y) in relator 2
        10: count of -2 (y^-1) in relator 2
        11: exponent_sum_x (net count of x across both relators)
        12: length_ratio (len_r1 / total_length), 0.5 means balanced
        13: max_min_ratio (max(len_r1, len_r2) / min(len_r1, len_r2))
    """
    presentation = np.asarray(presentation, dtype=np.int8)

    r1 = presentation[:max_relator_length]
    r2 = presentation[max_relator_length:]

    len_r1 = int(np.count_nonzero(r1))
    len_r2 = int(np.count_nonzero(r2))
    total_length = len_r1 + len_r2

    # Character counts per relator
    r1_nonzero = r1[:len_r1]
    r2_nonzero = r2[:len_r2]

    count_x_r1 = int(np.sum(r1_nonzero == 1))
    count_xinv_r1 = int(np.sum(r1_nonzero == -1))
    count_y_r1 = int(np.sum(r1_nonzero == 2))
    count_yinv_r1 = int(np.sum(r1_nonzero == -2))

    count_x_r2 = int(np.sum(r2_nonzero == 1))
    count_xinv_r2 = int(np.sum(r2_nonzero == -1))
    count_y_r2 = int(np.sum(r2_nonzero == 2))
    count_yinv_r2 = int(np.sum(r2_nonzero == -2))

    # Exponent sum of x across both relators
    exponent_sum_x = (count_x_r1 - count_xinv_r1) + (count_x_r2 - count_xinv_r2)

    # Length ratio
    length_ratio = len_r1 / total_length if total_length > 0 else 0.5

    # Max/min ratio
    min_len = min(len_r1, len_r2)
    max_len = max(len_r1, len_r2)
    max_min_ratio = max_len / min_len if min_len > 0 else max_len

    features = np.array([
        total_length,
        len_r1,
        len_r2,
        count_x_r1,
        count_xinv_r1,
        count_y_r1,
        count_yinv_r1,
        count_x_r2,
        count_xinv_r2,
        count_y_r2,
        count_yinv_r2,
        exponent_sum_x,
        length_ratio,
        max_min_ratio,
    ], dtype=np.float32)

    return features
