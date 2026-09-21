# -*- coding: utf-8 -*-
"""FIX D1. Rebuild hybrid height and diameter at the plot level.

Defect: 221_parse_hybrid_reps.py read each VNIIMK workbook column to the end of the
sheet, but the «Биометрия 2022/2023» sheets consist of nine blocks stacked one
under another. As a result 221_hybrid_plot_level.parquet received 159 records for
only 20 combinations (all VK101A), and each of them was a mean over several
foreign combinations at once.

Fix: height and diameter are taken from the already parsed plot-level raw table
FINAL_rebutS\02_ДАННЫЕ\Фенотипы.xlsx, sheet «Гибриды_делянки» (a row is a plant
within a plot), and collapsed to plot means. Seed yield, oil content, and
1000-seed weight are left untouched: they were fully reconciled with the VNIIMK
workbook (2882 values, zero discrepancies).

Outlier rejection for the new traits uses the same procedure as the original
pipeline (OLS residuals of Year + Mother + Father + Year:Rep, threshold |z| > 4),
so the fix has a single cause.

Input:  deposit 221_hybrid_plot_level.parquet   (productivity)
       FINAL_rebutS/02_ДАННЫЕ/Фенотипы.xlsx             (plant-level biometry)
Output: 221_hybrid_plot_level.parquet
       221_biometry_removed_outliers.csv
       fix01_biometry_summary.txt
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
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf

HERE = Path(__file__).resolve().parent
REPO = work()
SRC_PARQ = OUT / "221_hybrid_plot_level.parquet"
SRC_XLSX = REPO / "FINAL_rebutS" / "02_ДАННЫЕ" / "Фенотипы.xlsx"
OUT_PARQ = OUT / "221_hybrid_plot_level.parquet"
OUT_LOG  = OUT / "221_biometry_removed_outliers.csv"
OUT_TXT  = OUT / "fix01_biometry_summary.txt"
MOTHER = {"M1": "VK101A", "M2": "VA761A", "M3": "VK934A"}


def qc_outliers(d, thr=4.0):
    """Outlier rejection as in 221_parse_hybrid_reps.py: |z| of OLS residuals > 4."""
    x = d.copy()
    x["_y"] = x["value"].astype(float)
    x["_g"] = x["year"].astype(str); x["_m"] = x["mother"]; x["_f"] = x["father"]
    x["_r"] = x["_g"] + "_" + x["rep"].astype(str)
    fit = smf.ols("_y ~ C(_g) + C(_m) + C(_f) + C(_r)", data=x).fit()
    r = fit.resid
    z = (r - r.mean()) / r.std(ddof=1)
    return x.index[np.abs(z) > thr], z


H = en_table(pd.read_parquet(SRC_PARQ))
prod = H[H["trait"].isin(["seed_yield", "oil_content", "seed_weight_1000"])].copy()

S = en_table(pd.read_excel(SRC_XLSX, sheet_name="Гибриды_делянки"))
S = S[S["trait"].isin(["plant_height", "head_diameter"])].dropna(subset=["value"])
S["mother"] = S["mother"].map(MOTHER).fillna(S["mother"])
plots = (S.groupby(["mother", "line", "year", "replicate", "trait"])["value"]
           .agg(value="mean", plants="size").reset_index()
           .rename(columns={"line": "father", "year": "year",
                            "replicate": "rep", "trait": "trait"}))

removed, keep_parts = [], []
for tr, d in plots.groupby("trait"):
    d = d.reset_index(drop=True)
    idx, z = qc_outliers(d)
    if len(idx):
        rr = d.loc[idx].copy(); rr["z"] = z.loc[idx].values; removed.append(rr)
    keep_parts.append(d.drop(index=idx))
bio = pd.concat(keep_parts, ignore_index=True)
rem = (pd.concat(removed, ignore_index=True) if removed
       else pd.DataFrame(columns=list(plots.columns) + ["z"]))
rem.to_csv(OUT_LOG, index=False, encoding="utf-8-sig")

FIXED = pd.concat([prod, bio[list(prod.columns)]], ignore_index=True)
for c in ("year", "rep"):
    FIXED[c] = FIXED[c].astype(int)
FIXED = FIXED.sort_values(["trait", "year", "mother", "father", "rep"]).reset_index(drop=True)
FIXED.to_parquet(OUT_PARQ, index=False)

old_bio = H[H["trait"].isin(["plant_height", "head_diameter"])]
lines = []
w = lines.append
w("FIX D1 — hybrid height and diameter at the plot level")
w("")
w("before (221_hybrid_plot_level.parquet):")
w(f"  records {len(old_bio)}, combinations {old_bio.groupby(['mother','father']).ngroups}, "
  f"mothers {old_bio['mother'].nunique()}")
w(f"  duplicate keys Mother×Father×Year×Rep×Trait: "
  f"{int(old_bio.duplicated(['mother','father','year','rep','trait']).sum())}")
w("")
w("after (221_hybrid_plot_level.parquet):")
w(f"  records {len(bio)}, combinations {bio.groupby(['mother','father']).ngroups}, "
  f"mothers {bio['mother'].nunique()}")
w(f"  duplicate keys: {int(bio.duplicated(['mother','father','year','rep','trait']).sum())}")
w(f"  plots per combination×year×trait: "
  f"{dict(sorted(bio.groupby(['mother','father','year','trait']).size().value_counts().items()))}")
w(f"  plants per plot: min {int(bio['plants'].min())}, "
  f"median {int(bio['plants'].median())}, max {int(bio['plants'].max())}")
w(f"  outliers rejected: {len(rem)}")
w("")
w("final table:")
w(f"  total records {len(FIXED)} (was {len(H)})")
for tr, n in FIXED.groupby("trait").size().items():
    w(f"    {tr}: {n}")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
