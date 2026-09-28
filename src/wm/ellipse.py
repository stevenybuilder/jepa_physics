"""Circle, ellipse or bent line: shape estimators for the direction ring (scripts/run_ellipse.py).

All angles in radians. theta = the true direction of each centroid; P = centroids in a 2-D plane.
Three independent axis-ratio (q = b/a) estimators:
  1. shape fits in the centroid plane: Fitzgibbon direct least-squares conic, a polar fit
     r(phi) = a b / sqrt((b cos(phi - phi0))^2 + (a sin(phi - phi0))^2) about a free centre, and the supervised chart
     P ~ c + A [cos theta, sin theta] (singular values of A);
  2. full-space spectrum: an ellipse sampled evenly in theta has centroid-covariance eigenvalues a^2/2, b^2/2, so
     q = sqrt(lambda2 / lambda1); and its radius r(theta) puts its only non-constant r^2 power at 2 theta;
  3. angle distortion: atan2 on an ellipse is arctan(q tan u); its first harmonic has amplitude (1 - q) / (1 + q).
"""
import numpy as np
from scipy.optimize import least_squares

from wm.manifold import wrap_pi


# ---------------------------------------------------------------- 1. shape fits in a plane -------------------------

def conic_fit(P):
    """Fitzgibbon et al. 1999 direct least-squares ellipse (Halir & Flusser 1998 stable form). Always an ellipse.
    Returns dict(center [2], a, b (a >= b), q, ecc, angle (major axis, rad in [0, pi)))."""
    x, y = np.asarray(P, float).T
    s = max(x.std(), y.std())
    mx, my = x.mean(), y.mean()
    x, y = (x - mx) / s, (y - my) / s                            # conditioning; undone below
    D1 = np.stack([x * x, x * y, y * y], 1)
    D2 = np.stack([x, y, np.ones_like(x)], 1)
    S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2
    T = -np.linalg.solve(S3, S2.T)
    M = S1 + S2 @ T
    M = np.array([M[2] / 2, -M[1], M[0] / 2])
    w, V = np.linalg.eig(M)
    V = np.real(V)
    cond = 4 * V[0] * V[2] - V[1] ** 2
    a1 = V[:, np.argmax(cond)]
    A, B, C = a1
    Dd, E, F = T @ a1
    Mq = np.array([[A, B / 2], [B / 2, C]])
    g = np.array([Dd, E])
    x0 = -0.5 * np.linalg.solve(Mq, g)
    Fp = F + 0.5 * g @ x0
    lam, vec = np.linalg.eigh(Mq)
    axes = np.sqrt(-Fp / lam)                                    # eigh: ascending lam -> descending axes
    i = int(np.argmax(axes))
    a, b = axes[i] * s, axes[1 - i] * s
    ang = float(np.arctan2(vec[1, i], vec[0, i]) % np.pi)
    return {"center": (x0 * s + [mx, my]).tolist(), "a": float(a), "b": float(b), "q": float(b / a),
            "ecc": float(np.sqrt(max(1 - (b / a) ** 2, 0.0))), "angle": ang}


def ellipse_points(center, a, b, angle, n=4000):
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return np.asarray(center) + (np.stack([a * np.cos(u), b * np.sin(u)], 1) @ R.T)


def curve_distance(P, dense):
    """Distance from each row of P to a densely sampled closed curve."""
    d2 = ((np.asarray(P)[:, None, :] - dense[None]) ** 2).sum(-1)
    return np.sqrt(d2.min(1))


def circle_fit(P):
    """Geometric least-squares circle (Kasa start). Returns dict(center, R) and per-point radial residuals."""
    P = np.asarray(P, float)
    x, y = P.T
    M = np.stack([x, y, np.ones_like(x)], 1)
    c = np.linalg.lstsq(M, x * x + y * y, rcond=None)[0]
    cx, cy = c[0] / 2, c[1] / 2
    R0 = np.sqrt(c[2] + cx ** 2 + cy ** 2)
    res = least_squares(lambda p: np.hypot(x - p[0], y - p[1]) - p[2], [cx, cy, R0])
    return {"center": res.x[:2].tolist(), "R": float(res.x[2]), "resid": res.fun}


def polar_r(phi, a, b, phi0):
    """Radius of a centred ellipse at polar angle phi (major axis at phi0)."""
    return a * b / np.sqrt((b * np.cos(phi - phi0)) ** 2 + (a * np.sin(phi - phi0)) ** 2)


def polar_fit(P, init=None):
    """Direct fit of r(phi) = a b / sqrt((b cos(phi - phi0))^2 + (a sin(phi - phi0))^2) about a free centre
    (cx, cy, a, b, phi0). init: a conic_fit dict. Returns dict(center, a, b, q, ecc, angle, rms_radial)."""
    P = np.asarray(P, float)
    init = init or conic_fit(P)
    p0 = [*init["center"], init["a"], init["b"], init["angle"]]

    def f(p):
        d = P - p[:2]
        return np.hypot(d[:, 0], d[:, 1]) - polar_r(np.arctan2(d[:, 1], d[:, 0]), p[2], p[3], p[4])

    r = least_squares(f, p0)
    cx, cy, a, b, phi0 = r.x
    if abs(b) > abs(a):
        a, b, phi0 = b, a, phi0 + np.pi / 2
    a, b = abs(a), abs(b)
    return {"center": [float(cx), float(cy)], "a": float(a), "b": float(b), "q": float(b / a),
            "ecc": float(np.sqrt(max(1 - (b / a) ** 2, 0.0))), "angle": float(phi0 % np.pi),
            "rms_radial": float(np.sqrt(np.mean(r.fun ** 2)))}


def chart_fit(P, theta):
    """Supervised linear chart P ~ c + A [cos theta, sin theta] (an ellipse traversed at the true angle).
    Returns dict(c, A, a, b, q, pred [m, 2], angle of major axis)."""
    P, th = np.asarray(P, float), np.asarray(theta, float)
    M = np.stack([np.ones_like(th), np.cos(th), np.sin(th)], 1)
    coef = np.linalg.lstsq(M, P, rcond=None)[0]
    c, A = coef[0], coef[1:].T                                   # A [q, 2]
    U, S, _ = np.linalg.svd(A)
    return {"c": c, "A": A, "a": float(S[0]), "b": float(S[1]), "q": float(S[1] / S[0]), "pred": M @ coef,
            "angle": float(np.arctan2(U[1, 0], U[0, 0]) % np.pi) if A.shape[0] == 2 else None}


def circle_chart_fit(P, theta):
    """Supervised circle at the true angle: P ~ c + R Rot [cos(s theta), sin(s theta)] (best orientation s).
    Returns the predictions [m, 2]."""
    P, th = np.asarray(P, float), np.asarray(theta, float)
    best = None
    for s in (1, -1):
        T = np.stack([np.cos(s * th), np.sin(s * th)], 1)
        Pc, Tc = P - P.mean(0), T - T.mean(0)
        U, S, Vt = np.linalg.svd(Pc.T @ Tc)
        R = U @ Vt
        scale = S.sum() / (Tc ** 2).sum()
        pred = P.mean(0) + scale * Tc @ R.T
        err = ((P - pred) ** 2).sum()
        if best is None or err < best[0]:
            best = (err, pred)
    return best[1]


def chart_curvature(A, theta):
    """Curvature of the chart ellipse c + A [cos t, sin t] at t = theta (any ambient dimension)."""
    th = np.atleast_1d(np.asarray(theta, float))
    d1 = np.stack([-np.sin(th), np.cos(th)], 1) @ A.T
    d2 = -np.stack([np.cos(th), np.sin(th)], 1) @ A.T
    return curve_curvature(d1, d2)


def curve_curvature(d1, d2):
    """kappa = sqrt(|r'|^2 |r''|^2 - (r'.r'')^2) / |r'|^3, rows = points."""
    n1, n2, dot = (d1 ** 2).sum(-1), (d2 ** 2).sum(-1), (d1 * d2).sum(-1)
    return np.sqrt(np.maximum(n1 * n2 - dot ** 2, 0)) / n1 ** 1.5


# ---------------------------------------------------------------- 2. full-space spectrum ---------------------------

def centroid_spectrum(C, noise_cov_trace=None, noise_cov=None, top=6):
    """Eigenvalues of the centroid covariance (C [m, D], divided by m), optionally minus the centroid-noise
    covariance noise_cov [D, D] (= within-value covariance times mean(1 / count)). Returns raw and corrected
    q = sqrt(lambda2 / lambda1) and the variance fraction outside the top-2 plane."""
    Cc = np.asarray(C, float) - np.asarray(C, float).mean(0)
    m = len(Cc)
    lam = np.linalg.svd(Cc, compute_uv=False) ** 2 / m
    out = {"eig_top": lam[:top].tolist(), "q_eig": float(np.sqrt(lam[1] / lam[0])),
           "frac_outside_top2": float(1 - lam[:2].sum() / lam.sum())}
    if noise_cov is not None:
        cov = Cc.T @ Cc / m - noise_cov * (m - 1) / m
        lc = np.sort(np.linalg.eigvalsh(cov))[::-1]
        pos = np.clip(lc, 0, None)
        out.update({"eig_top_noise_corrected": lc[:top].tolist(),
                    "q_eig_noise_corrected": float(np.sqrt(max(lc[1], 0) / lc[0])),
                    "frac_outside_top2_noise_corrected": float(1 - pos[:2].sum() / pos.sum()),
                    "noise_trace_frac_of_centroid_var": float(np.trace(noise_cov) * (m - 1) / m / lam.sum())})
    return out


def radius_fourier(C, theta, noise_r2=None, kmax=8):
    """r(theta) = |C_i - mean C| (any dimension). Power P_k = |F_k|^2 (x2 for k >= 1), F_k = mean r e^{-ik theta}.
    Returns P2/P0, the 2-theta share of the k >= 1 power, and q from r^2 (an ellipse has
    r^2 = (a^2 + b^2)/2 + (a^2 - b^2)/2 cos 2(theta - psi)): q = sqrt((1 - rho) / (1 + rho)),
    rho = 2 |F_2(r^2)| / F_0(r^2). noise_r2 [m]: expected ||centroid noise||^2 per centroid, subtracted from r^2."""
    Cc = np.asarray(C, float) - np.asarray(C, float).mean(0)
    th = np.asarray(theta, float)
    r2 = (Cc ** 2).sum(1)
    r = np.sqrt(r2)
    k = np.arange(kmax + 1)
    F = np.exp(-1j * np.outer(k, th)) @ r / len(r)
    Pk = np.abs(F) ** 2 * np.where(k > 0, 2, 1)

    def q_from(r2v):
        G = np.exp(-1j * np.array([0, 2])[:, None] * th[None]) @ r2v / len(r2v)
        rho = min(2 * abs(G[1]) / G[0].real, 1.0)
        return float(np.sqrt((1 - rho) / (1 + rho)))

    out = {"power_k": Pk.tolist(), "p2_over_p0": float(Pk[2] / Pk[0]),
           "share_2theta_of_k_ge_1": float(Pk[2] / Pk[1:].sum()), "q_from_r2": q_from(r2)}
    if noise_r2 is not None:
        out["q_from_r2_noise_corrected"] = q_from(r2 - np.asarray(noise_r2, float))
    return out


# ---------------------------------------------------------------- 3. angle distortion ------------------------------

def angle_distortion(phi, theta):
    """phi: unsupervised angle atan2(PC2, PC1) of each centroid; theta: true angle. Aligns orientation s and offset,
    then fits (a) residual = c + alpha sin 2theta + beta cos 2theta (R^2, amplitude eps -> q = (1 - eps)/(1 + eps))
    and (b) the exact model phi = psi + arg(cos u + i q sin u), u = s theta + delta (q, psi, delta free)."""
    phi, th = np.asarray(phi, float), np.asarray(theta, float)
    best = None
    for s in (1, -1):
        off = np.angle(np.exp(1j * (phi - s * th)).mean())
        d = wrap_pi(phi - s * th - off)
        if best is None or np.abs(d).mean() < np.abs(best[1]).mean():
            best = (s, d, off)
    s, d, off = best
    M = np.stack([np.ones_like(th), np.sin(2 * th), np.cos(2 * th)], 1)
    coef = np.linalg.lstsq(M, d, rcond=None)[0]
    fit = M @ coef
    ss = ((d - d.mean()) ** 2).sum()
    r2 = float(1 - ((d - fit) ** 2).sum() / ss) if ss > 0 else float("nan")
    eps = float(np.hypot(coef[1], coef[2]))
    q_lin = (1 - eps) / (1 + eps)
    # exact model: start delta from the 2-theta phase (d ~ -eps sin 2u, u = s theta + delta), psi = off - delta
    d0 = np.arctan2(-coef[2], -s * coef[1]) / 2 if eps > 0 else 0.0

    def model(p):
        q, psi, delta = p
        u = s * th + delta
        return wrap_pi(psi + np.arctan2(q * np.sin(u), np.cos(u)) - phi)

    q0 = float(np.clip(q_lin, 0.01, 0.99))
    starts = [[q0, off - d0, d0], [q0, off - d0 - np.pi / 2, d0 + np.pi / 2], [q0, off, 0.0]]
    fits = [least_squares(model, p0, bounds=([1e-3, -np.inf, -np.inf], [1.0, np.inf, np.inf])) for p0 in starts]
    ex = min(fits, key=lambda r: r.cost)
    return {"orientation": int(s), "offset_rad": float(off), "resid_rms_deg": float(np.degrees(np.sqrt(np.mean(d ** 2)))),
            "mean_abs_dev_deg": float(np.degrees(np.abs(d).mean())), "eps_2theta": eps, "r2_2theta": r2,
            "q_linear": float(q_lin), "q_exact": float(ex.x[0]),
            "exact_resid_rms_deg": float(np.degrees(np.sqrt(np.mean(ex.fun ** 2))))}


# ---------------------------------------------------------------- non-planar rings ---------------------------------

def harmonic_variance(C, theta, noise_trace=None, kmax=8):
    """Centroid variance by Fourier harmonic of theta: C(theta) = mu + sum_k A_k cos k theta + B_k sin k theta, over
    all D dimensions; var_k = (|A_k|^2 + |B_k|^2) / 2. An ellipse (any linear image of a circle) is pure k = 1; a ring
    bent out of its plane by a saddle (z ~ cos 2 theta) adds k = 2. theta must be an even grid (the 64 values).
    noise_trace: trace of the centroid-noise covariance; each harmonic pair carries 2 / m of it, subtracted."""
    Cc = np.asarray(C, float) - np.asarray(C, float).mean(0)
    th = np.asarray(theta, float)
    m = len(Cc)
    tot = (Cc ** 2).sum() / m
    k = np.arange(1, kmax + 1)
    F = np.exp(-1j * np.outer(k, th)) @ Cc / m                    # [kmax, D]
    var = 2 * (np.abs(F) ** 2).sum(1)
    out = {"k": k.tolist(), "var": var.tolist(), "share": (var / tot).tolist(),
           "share_k_gt_kmax": float(1 - var.sum() / tot), "bend_ratio_k2_over_k1": float(np.sqrt(var[1] / var[0]))}
    if noise_trace is not None:
        noise = 2 * noise_trace / m
        vc = np.clip(var - noise, 0, None)
        totc = tot - (m - 1) / m * noise_trace
        out.update({"share_noise_corrected": (vc / totc).tolist(),
                    "bend_ratio_k2_over_k1_noise_corrected": float(np.sqrt(vc[1] / vc[0]))})
    return out


def chart_q_full(C, theta, noise_trace=None):
    """Axis ratio of the k = 1 component in the full space: singular values of A in C ~ mu + A [cos, sin]. Noise in A
    adds about e = (2 / m) trace(centroid-noise covariance) to each squared singular value; subtracted when given."""
    ch = chart_fit(C, theta)
    s1, s2 = ch["a"], ch["b"]
    out = {"a": s1, "b": s2, "q": s2 / s1}
    if noise_trace is not None:
        e = 2 * noise_trace / len(np.asarray(C))
        out["q_noise_corrected"] = float(np.sqrt(max(s2 ** 2 - e, 0) / max(s1 ** 2 - e, 1e-300)))
        out["noise_e_over_b2"] = float(e / s2 ** 2)
    return out, ch
