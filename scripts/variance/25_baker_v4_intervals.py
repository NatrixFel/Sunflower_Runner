# -*- coding: utf-8 -*-
"""B4 intervals. Profile likelihood interval for Baker's ratio
and standard errors of variance-component shares (whiskers for Fig. 1B).

The baker_v3.py bootstrap is invalid: it relabels every resampled combination
as a distinct “father”, destroying the factorial structure of the design.
Here a profile likelihood on the ratio itself is used instead.

The ratio b = 2 gamma_L / (2 gamma_L + gamma_ML) depends only on gamma,
so at fixed b we substitute gamma_ML = 2 gamma_L (1-b) / b
and maximize the likelihood over the rest. The interval boundary is
2 (logL_max - logL_profile) = 3.841.

The component share share_i = gamma_i / (sum gamma + 1) also depends only
on gamma, so its standard error comes from the delta method
on the same Hessian without involving sigma2_e.
"""
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import sys, io, json, pathlib
import numpy as np
from scipy import optimize, stats
import warnings; warnings.filterwarnings("ignore")
from baker_v4 import (REML, dummies, inter, build, load_repo, ROOT)

NAMES = ["line", "tester_x_line", "year_x_line", "year_x_tester_x_line"]
STARTS = [[.2, .05, .3, .3], [.05, .05, .05, .05], [1., .2, 1., 1.],
          [.4, .01, .4, .05], [.2, .05, 1e-9, .5]]
CRIT = stats.chi2.ppf(0.95, 1)          # 3.8415


def model(d):
    """Specification adopted by the author: Year × Tester is in X as a fixed effect."""
    y, Y, R, M, L = build(d)
    yr = d["year"].astype(str); mo = d["mother"].astype(str); fa = d["line"].astype(str)
    X = np.hstack([np.ones((len(y), 1)), M, R, inter(Y, M)])
    Zs = [dummies(fa.tolist()), dummies((mo + "_" + fa).tolist()),
          dummies((yr + "_" + fa).tolist()), dummies((yr + "_" + mo + "_" + fa).tolist())]
    return REML(y, X, Zs, NAMES)


def profile_negll(m, b, g0):
    """min negll at a fixed Baker's ratio b.

    L-BFGS-B over three free parameters with bounds; several starts,
    because the surface is flat at the boundary."""
    if b <= 0.0:
        f = lambda v: m.negll([0.0, v[0], v[1], v[2]])
        starts = [[max(g0[1], 1e-3), g0[2], g0[3]], [1e-2, g0[2], g0[3]]]
        bnd = [(0.0, 1e4)] * 3
    else:
        k = 2.0 * (1.0 - b) / b          # gamma_ML = k * gamma_L
        f = lambda v: m.negll([v[0], k * v[0], v[1], v[2]])
        starts = [[max(g0[0], 1e-4), g0[2], g0[3]],
                  [1e-3, g0[2], g0[3]],
                  [max(g0[0], 1e-4) * 3, g0[2], g0[3]]]
        bnd = [(0.0, 1e4)] * 3
    best = np.inf
    for st in starts:
        r = optimize.minimize(f, st, method="L-BFGS-B", bounds=bnd,
                              options=dict(maxiter=300, ftol=1e-14, gtol=1e-10))
        best = min(best, float(r.fun))
    return best


def profile_interval(m, fit, grid_step=0.05):
    g0 = fit["gamma"]; f0 = -fit["logL"]
    bhat = 2 * g0[0] / (2 * g0[0] + g0[1])
    grid = np.round(np.arange(0.0, 1.0 + 1e-9, grid_step), 6)
    prof = {}
    for b in grid:
        prof[float(b)] = 2.0 * (profile_negll(m, float(b), g0) - f0)

    def dev(b):
        b = float(np.clip(b, 0.0, 1.0))
        if b in prof: return prof[b]
        v = 2.0 * (profile_negll(m, b, g0) - f0)
        prof[b] = v
        return v

    def bisect(lo, hi):
        """lo inside (dev<=CRIT), hi outside; return the boundary."""
        for _ in range(16):
            mid = 0.5 * (lo + hi)
            if dev(mid) <= CRIT: lo = mid
            else: hi = mid
        return 0.5 * (lo + hi)

    inside = [b for b in grid if prof[float(b)] <= CRIT]
    lo_b = min(inside); hi_b = max(inside)
    low = 0.0 if lo_b <= 1e-12 else bisect(lo_b, lo_b - grid_step)
    high = 1.0 if hi_b >= 1.0 - 1e-12 else bisect(hi_b, hi_b + grid_step)
    return dict(bhat=float(bhat), low=float(low), high=float(high),
                profile={f"{k:.4f}": float(v) for k, v in sorted(prof.items())})


def share_se(m, fit):
    """Shares of the five components in their sum, and their SEs (delta method).
    share_i = gamma_i / (sum gamma + 1); error share = 1 / (sum gamma + 1)."""
    g = np.asarray(fit["gamma"], float)
    H = m.hessian(g)
    C = np.linalg.inv(H)
    S = g.sum() + 1.0
    shares = np.append(g / S, 1.0 / S)
    J = np.zeros((5, 4))
    for i in range(4):
        for j in range(4):
            J[i, j] = ((1.0 if i == j else 0.0) * S - g[i]) / S ** 2
    J[4, :] = -1.0 / S ** 2
    se = np.sqrt(np.clip(np.diag(J @ C @ J.T), 0, None))
    return shares, se


if __name__ == "__main__":
    repo = load_repo()
    out = {}
    for key, name in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"),
                      ("seed_weight_1000", "seed_weight_1000")]:
        d = repo[repo.trait == key].reset_index(drop=True)
        m = model(d)
        fit = m.fit(STARTS)
        s2 = fit["s2"]
        bhat = 2 * s2["line"] / (2 * s2["line"] + s2["tester_x_line"])
        ci = profile_interval(m, fit)
        sh, se = share_se(m, fit)
        print("\n" + "=" * 88)
        print(f"### {name}")
        print(f"  Baker's ratio = {bhat:.4f};  95% profile interval = "
              f"{ci['low']:.3f}–{ci['high']:.3f}")
        lab = NAMES + ["error"]
        for i, l in enumerate(lab):
            print(f"      share {l:18s} = {100*sh[i]:6.2f} % ± {100*se[i]:.2f} %")
        print(f"      sum of shares = {sh.sum():.6f}")
        out[key] = dict(name=name, baker=float(bhat), ci_low=ci["low"], ci_high=ci["high"],
                        shares=[float(x) for x in sh], share_se=[float(x) for x in se],
                        labels=lab, s2={k: float(v) for k, v in s2.items()},
                        s2e=float(fit["s2e"]), gamma=[float(x) for x in fit["gamma"]],
                        profile=ci["profile"])
    (ROOT / "brain" / "25_baker_v4_intervals.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved brain\\25_baker_v4_intervals.json")
