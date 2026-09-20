"""Compact EMMAX (Kang et al. 2010) — for comparison with BLINK. Ported from script 13/18."""
from __future__ import annotations

import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import numpy as np
from scipy import stats, optimize


def vanraden_K(G_imp: np.ndarray) -> np.ndarray:
    """K-matrix VanRaden. G_imp: (m, n) imputed ALT dosage."""
    af = G_imp.mean(axis=1) / 2
    W = (G_imp.T - 2 * af)
    denom = 2 * np.sum(af * (1 - af))
    return W @ W.T / denom


def _reml_loglik(theta_log, y_rot, X_rot, eigvals):
    sig2g = np.exp(theta_log[0]); sig2e = np.exp(theta_log[1])
    D = sig2g * eigvals + sig2e
    if (D <= 0).any(): return 1e10
    W = 1.0 / D
    XtWX = X_rot.T @ (X_rot * W[:, None])
    try: XtWX_inv = np.linalg.inv(XtWX)
    except np.linalg.LinAlgError: return 1e10
    beta = XtWX_inv @ (X_rot.T @ (y_rot * W))
    resid = y_rot - X_rot @ beta
    rss = (resid ** 2 * W).sum()
    _, log_det_XtWX = np.linalg.slogdet(XtWX)
    return 0.5 * (np.log(D).sum() + rss + log_det_XtWX)


def emmax_run(y: np.ndarray, K: np.ndarray, G: np.ndarray, PCs: np.ndarray | None = None):
    """EMMAX GWAS. y(n,), K(n,n), G(m,n). PCs(n,p) fixed covariates. Returns pvals, betas, lambda, h2."""
    mask = ~np.isnan(y)
    y_v = y[mask]; K_sub = K[np.ix_(mask, mask)]; G_sub = G[:, mask]
    eigvals, U = np.linalg.eigh(K_sub)
    eigvals = np.maximum(eigvals, 1e-9)
    y_rot = U.T @ y_v
    X_const = np.ones((len(y_v), 1))
    if PCs is not None and PCs.shape[1] > 0:
        X_const = np.column_stack([X_const, PCs[mask]])
    X_rot = U.T @ X_const
    var_y = np.var(y_v)
    res = optimize.minimize(_reml_loglik, np.log([0.5*var_y, 0.5*var_y]),
                            args=(y_rot, X_rot, eigvals), method="L-BFGS-B",
                            bounds=[(np.log(1e-8), np.log(1e8))]*2)
    sig2g, sig2e = np.exp(res.x)
    weights = 1.0 / (sig2g * eigvals + sig2e)
    sqrt_w = np.sqrt(weights)
    Xw = X_rot * sqrt_w[:, None]; y_w = y_rot * sqrt_w
    df = len(y_v) - X_const.shape[1] - 1
    m = G_sub.shape[0]
    pvals = np.full(m, np.nan); betas = np.full(m, np.nan)
    for i in range(m):
        g = G_sub[i]
        if np.std(g) < 1e-6: continue
        g_rot = (U.T @ g) * sqrt_w
        Xf = np.column_stack([Xw, g_rot])
        try:
            beta, *_ = np.linalg.lstsq(Xf, y_w, rcond=None)
            resid = y_w - Xf @ beta
            mse = (resid**2).sum() / df
            se = np.sqrt(mse * np.linalg.inv(Xf.T @ Xf)[-1, -1])
            if se > 0:
                t = beta[-1] / se
                pvals[i] = 2 * stats.t.sf(abs(t), df); betas[i] = beta[-1]
        except (np.linalg.LinAlgError, ValueError): continue
    valid = ~np.isnan(pvals)
    lam = (np.median(stats.chi2.isf(pvals[valid], 1)) / stats.chi2.ppf(0.5, 1)
           if valid.sum() else np.nan)
    return pvals, betas, float(lam), float(sig2g / (sig2g + sig2e))
