"""Planted-code synthetic data shared by the probe, INLP and steering tests."""
import numpy as np


def copies_code(y_cols, n_copies, d=40, seed=0, noise0=0.1, ratio=1.6):
    """X [N, d] carrying the target columns y_cols [N, m] in n_copies orthogonal m-dim slots, copy j
    with independent noise of SD noise0·ratio^j, plus unit noise in the other d − m·n_copies dims,
    all rotated by a random orthogonal matrix. Every copy alone decodes y, so the linear information
    has exactly m·n_copies orthogonal carriers; the distinct noise levels make each probe of the
    sequence lock onto one copy."""
    rng = np.random.default_rng(seed)
    n, m = y_cols.shape
    slots = [y_cols + noise0 * ratio ** j * rng.standard_normal((n, m)) for j in range(n_copies)]
    Z = np.hstack(slots + [rng.standard_normal((n, d - m * n_copies))])
    rotation = np.linalg.qr(rng.standard_normal((d, d)))[0]
    return Z @ rotation.T


def split(n, seed=0):
    """Random 80/20 train/test mask and 5 folds on train."""
    rng = np.random.default_rng(seed)
    test = rng.random(n) < 0.2
    return ~test, test, rng.integers(0, 5, (~test).sum())
