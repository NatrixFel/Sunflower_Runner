# -*- coding: utf-8 -*-
"""
Year-to-year repeatability of general and specific combining ability.

Paternal GCA is estimated separately from 2022 and 2023 plot-level data, then
the two estimates are compared by Pearson correlation; the interval uses
Fisher’s transformation. The quantity answers whether an effect estimated
in one year is reproduced in the other.
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
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

D = DEPOSIT
plot = en_table(pd.read_parquet(OUT / "22_hybrid_plot_level.parquet"))
CONF = ("LI29", "LI30")


def gca_by_year(d):
    out = {}
    for yr in sorted(d["year"].unique()):
        tab = d[d["year"] == yr].groupby(["father", "mother"])["value"].mean().unstack()
        m = tab.mean(1)
        out[yr] = m - m.mean()
    return out


def sca_by_year(d):
    out = {}
    for yr in sorted(d["year"].unique()):
        tab = d[d["year"] == yr].groupby(["father", "mother"])["value"].mean().unstack()
        g = np.nanmean(tab.values)
        gf = np.nanmean(tab.values, 1) - g
        gm = np.nanmean(tab.values, 0) - g
        out[yr] = pd.DataFrame(tab.values - g - gf[:, None] - gm[None, :],
                               index=tab.index, columns=tab.columns).stack()
    return out


def ci(r, n):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    return np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)


print(f"{'trait':18s} {'panel':14s} {'r':>8s} {'p':>9s} {'95 % CI':>18s}  n")
for tr, en in [("seed_yield", "seed yield"), ("oil_content", "oil content"),
               ("seed_weight_1000", "1000-seed weight")]:
    d = plot[plot["trait"] == tr].dropna(subset=["value"])
    g = gca_by_year(d)
    yrs = sorted(g)
    j = pd.concat([g[yrs[0]].rename("y1"), g[yrs[1]].rename("y2")], axis=1).dropna()
    for label, sub in [("54 lines", j), ("52 oilseed", j.drop(index=[c for c in CONF if c in j.index]))]:
        r, p = stats.pearsonr(sub.y1, sub.y2)
        lo, hi = ci(r, len(sub))
        print(f"{en:18s} {label:14s} {r:8.4f} {p:9.4f}   {lo:+.4f}…{hi:+.4f}  {len(sub)}")
    sc = sca_by_year(d)
    js = pd.concat([sc[yrs[0]].rename("y1"), sc[yrs[1]].rename("y2")], axis=1).dropna()
    r, p = stats.pearsonr(js.y1, js.y2)
    print(f"{'':18s} {'SCA, all pairs':14s} {r:8.4f} {p:9.4f} {'':>18s}  {len(js)}")
