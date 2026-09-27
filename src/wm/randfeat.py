"""Random-feature floor for the Part 1 layer curves (Part 1 extra, spec §6 item 2 / §7 outcomes).

A random-init ViT-L reads direction, speed and acceleration from block 1 about as well as V-JEPA 2. The question this
answers: is that more than "fixed random nonlinear features of a clean low-dimensional input"? The input is the disk
centroid trajectory of wm.pixels (x_t, y_t per frame / 256, 32 numbers) or its 15 frame differences (30 numbers),
z-scored on train (NaN -> train mean, wm.probes.Standardizer); the features are relu(Z W + b) with fixed Gaussian
W [d_in, 1024] (scale 1/sqrt(d_in)) and b ~ N(0, 1), then the identical ridge / α-CV / test-once of wm.pixels.probe_row.
"""
import numpy as np

from wm.data import SIZE
from wm.probes import Standardizer

N_FEATURES = 1024


def random_relu_features(Ztr, Zte, n_features=N_FEATURES, seed=0):
    """Fixed random ReLU features of standardised inputs, then z-scored with train statistics.
    Ztr [n_tr, d_in], Zte [n_te, d_in] -> (Ftr, Fte) float64 [n, n_features]."""
    d_in = Ztr.shape[1]
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((d_in, n_features)) / np.sqrt(d_in)
    b = rng.standard_normal(n_features)
    Ftr, Fte = np.maximum(Ztr @ W + b, 0.0), np.maximum(Zte @ W + b, 0.0)
    st = Standardizer().fit(Ftr)
    return st.transform(Ftr), st.transform(Fte)


def centroid_inputs(cen, tr, te, which="centroid"):
    """Train-standardised (Ztr, Zte) from centroids cen [N, 16, 2] px: 'centroid' = (x, y)/256 per frame (32),
    'diff' = its 15 frame differences (30)."""
    c = np.asarray(cen, float) / SIZE
    n = len(c)
    X = c.reshape(n, -1) if which == "centroid" else np.diff(c, axis=1).reshape(n, -1)
    st = Standardizer().fit(X[tr])
    return st.transform(X[tr]), st.transform(X[te])
