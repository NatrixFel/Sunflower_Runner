# -*- coding: utf-8 -*-
"""B4 check. Sensitivity of Baker's ratio to whether the
Year × Tester term is in the fixed-effects matrix.

Year and tester are declared fixed; therefore their interaction is fixed as well.
In brain\\BAKER_MODEL_AUDIT.md the headline numbers (0.4416 / 0.9591 / 0.9754) were
obtained WITHOUT it; scripts/baker_v4.py puts it in X. The script prints both
specifications side by side for the three traits.
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
import sys, io, pathlib
import numpy as np, pandas as pd
from scipy import stats
import warnings; warnings.filterwarnings("ignore")
from baker_v4 import (REML, dummies, inter, build, moments, load_repo)

NF = ["line", "tester_x_line", "year_x_line", "year_x_tester_x_line"]
NN = ["line", "tester_x_line", "year_x_tester_x_line"]
ST4 = [[.2, .05, .3, .3], [.05, .05, .05, .05], [1., .2, 1., 1.],
       [.4, .01, .4, .05], [.2, .05, 1e-9, .5]]
ST3 = [[.2, .05, .3], [.05, .05, .05], [1., .2, 1.], [.4, .01, .05]]


def fit(d, ym_fixed, with_yl):
    y, Y, R, M, L = build(d)
    yr = d["year"].astype(str); mo = d["mother"].astype(str); fa = d["line"].astype(str)
    parts = [np.ones((len(y), 1)), M, R]
    if ym_fixed:
        parts.append(inter(Y, M))
    X = np.hstack(parts)
    Zl = dummies(fa.tolist()); Zml = dummies((mo + "_" + fa).tolist())
    Zyl = dummies((yr + "_" + fa).tolist())
    Zyml = dummies((yr + "_" + mo + "_" + fa).tolist())
    Zs, names, st = ([Zl, Zml, Zyl, Zyml], NF, ST4) if with_yl else ([Zl, Zml, Zyml], NN, ST3)
    m = REML(y, X, Zs, names)
    f = m.fit(st)
    s2 = f["s2"]; den = 2 * s2["line"] + s2["tester_x_line"]
    f["baker"] = 2 * s2["line"] / den if den > 0 else float("nan")
    f["p"] = m.p; f["obj"] = m
    return f


if __name__ == "__main__":
    repo = load_repo()
    for key, name in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"),
                      ("seed_weight_1000", "seed_weight_1000")]:
        d = repo[repo.trait == key].reset_index(drop=True)
        A = moments(d)["anova"]
        print("\n" + "=" * 92)
        print(f"### {name}:  MS(year_x_mother) = {A['year_x_mother']['MS']:.4f} at df={A['year_x_mother']['df']}, "
              f"MS(year_x_mother_x_line) = {A['year_x_mother_x_line']['MS']:.4f}, "
              f"F = {A['year_x_mother']['MS']/A['year_x_mother_x_line']['MS']:.2f}, "
              f"p = {stats.f.sf(A['year_x_mother']['MS']/A['year_x_mother_x_line']['MS'], A['year_x_mother']['df'], A['year_x_mother_x_line']['df']):.3g}")
        for ym in (False, True):
            f = fit(d, ym, True)
            n = fit(d, ym, False)
            lr = 2 * (f["logL"] - n["logL"])
            pv = 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0
            lab = "Year×Tester IN X (fixed)" if ym else "Year×Tester NOT in X"
            g = f["gamma"]
            print(f"  {lab:32s} rank X={f['p']}  Baker's ratio={f['baker']:.4f}  "
                  f"logL={f['logL']:.4f}")
            print(f"      s2: Line={f['s2']['line']:.5f}  SCA={f['s2']['tester_x_line']:.5f}  "
                  f"G×L={f['s2']['year_x_line']:.5f}  G×T×L={f['s2']['year_x_tester_x_line']:.5f}  "
                  f"error={f['s2e']:.5f}")
            print(f"      gamma(G×L)={g[2]:.4e}{'   <-- BOUNDARY 0' if g[2] <= 1e-9 else ''}"
                  f"   LR for G×L = {lr:.4f}, p = {pv:.4g}")
