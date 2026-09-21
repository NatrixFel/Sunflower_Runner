# -*- coding: utf-8 -*-
"""R3.3. Year fixed or random in the BLUP model.

The production script `221_parse_hybrid_reps.py` fits
    value ~ C(year) + C(year):C(rep),  groups = hybrid
i.e. genotype random, year and replicate(year) FIXED.
The Methods 2.2 text describes something else: `Value ~ 1 + (1|Genotype) + (1|Year) + ε`.

Both specifications are fitted here on the current data:
  A (production): year and replicate(year) fixed, genotype random;
  B (as in the text): genotype and year random, intercept the only fixed effect.
B is fitted with a custom REML over three components, because with two year levels
standard wrappers become singular. Read-only.
"""
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
import sys, pathlib, warnings
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import optimize

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
D = DEPOSIT
plot = en_table(pd.read_parquet(OUT / '221_hybrid_plot_level.parquet'))
TR = ["seed_yield", "oil_content", "seed_weight_1000"]   # three traits for which the paper reports H²

def dummies(keys):
    lv = sorted(set(keys)); idx = {l: i for i, l in enumerate(lv)}
    Z = np.zeros((len(keys), len(lv)))
    for i, k in enumerate(keys):
        Z[i, idx[k]] = 1.0
    return Z, lv

def reml_two(y, X, Zs):
    """REML for y = Xb + sum Z_i u_i + e; returns components and the BLUP of the first group."""
    n = len(y)
    def negll(lv):
        v = np.exp(lv); s2e = v[-1]
        V = np.eye(n) * s2e
        for Z, s in zip(Zs, v[:-1]):
            V += s * (Z @ Z.T)
        try:
            L = np.linalg.cholesky(V)
        except np.linalg.LinAlgError:
            return 1e12
        Vi = np.linalg.inv(V)
        ldV = 2 * np.sum(np.log(np.diag(L)))
        XtVi = X.T @ Vi
        A = XtVi @ X
        sgn, ldA = np.linalg.slogdet(A)
        if sgn <= 0:
            return 1e12
        b = np.linalg.solve(A, XtVi @ y)
        r = y - X @ b
        return 0.5 * (ldV + ldA + r @ Vi @ r)
    v0 = np.log(np.full(len(Zs) + 1, np.var(y) / (len(Zs) + 1)))
    best = None
    for sc in (1.0, 0.3, 3.0):
        res = optimize.minimize(negll, v0 + np.log(sc), method='Nelder-Mead',
                                options=dict(maxiter=4000, xatol=1e-9, fatol=1e-9))
        if best is None or res.fun < best.fun:
            best = res
    v = np.exp(best.x); s2e = v[-1]
    V = np.eye(n) * s2e
    for Z, s in zip(Zs, v[:-1]):
        V += s * (Z @ Z.T)
    Vi = np.linalg.inv(V)
    b = np.linalg.solve(X.T @ Vi @ X, X.T @ Vi @ y)
    u = v[0] * (Zs[0].T @ (Vi @ (y - X @ b)))
    return v[:-1], s2e, u

print('%-13s %-24s %10s %10s %14s %9s' % ('trait', 'specification', 'σ²g', 'σ²e', 'repeatability', 'H²'))
print('-' * 88)
res = {}
for t in TR:
    d = plot[plot["trait"] == t].dropna(subset=["value"]).copy()
    d["g"] = d["mother"].astype(str) + "_" + d["father"].astype(str)
    nrep = d.groupby("g").size().mean()
    y = d["value"].to_numpy(float)

    # A — production
    a = smf.mixedlm("value ~ C(year)+C(year):C(rep)", d, groups=d["g"]).fit(reml=True)
    s2gA, s2eA = float(a.cov_re.iloc[0, 0]), float(a.scale)
    blupA = pd.Series({g: float(v.iloc[0]) for g, v in a.random_effects.items()})

    # B — as in the text: genotype and year random
    Zg, lg = dummies(d["g"].tolist())
    Zy, ly = dummies(d["year"].astype(str).tolist())
    X = np.ones((len(y), 1))
    (s2gB, s2yB), s2eB, uB = reml_two(y, X, [Zg, Zy])
    blupB = pd.Series(uB, index=lg)

    for nm, s2g, s2e in (('A: year fixed', s2gA, s2eA), ('B: year random', s2gB, s2eB)):
        print('%-13s %-24s %10.4f %10.4f %14.4f %9.4f'
              % (t, nm, s2g, s2e, s2g / (s2g + s2e), s2g / (s2g + s2e / nrep)))
    idx = blupA.index.intersection(blupB.index)
    r = np.corrcoef(blupA[idx], blupB[idx])[0, 1]
    print('%-13s %-24s σ²year = %.4f   r(BLUP A,B) = %.6f   max|Δ| = %.5f'
          % ('', 'check', s2yB, r, float(np.max(np.abs(blupA[idx] - blupB[idx])))))
    print()
    res[t] = dict(s2gA=s2gA, s2gB=s2gB, r=r)
