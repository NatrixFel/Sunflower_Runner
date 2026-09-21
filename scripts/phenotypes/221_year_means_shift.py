# -*- coding: utf-8 -*-
"""
A one-line shift in the year-means summary table and its correction.

What it does:
  1. Tests the shift hypothesis: for tester M1 in 2022, the value standing
     against line LI_i matches the raw value for line LI_i+1.
     The check runs over five traits and three testers, so it contains
     its own negative control.
  2. Compares plot-level raw data from two sources — `221_hybrid_plot_level.parquet`
     and the `Гибриды_делянки` sheet of `Фенотипы.xlsx` — and shows that on the
     three shared traits they are identical.
  3. Rebuilds the summary table for height, head diameter, and 1000-seed weight.
     Oil content is not rebuilt: its four discordant cells are outlier removals
     that the summary table does not carry, i.e. an explained difference.
     Seed yield agrees with the plot-level data to 0.0004 and is left untouched.
  4. Writes `221_hybrid_means_corrected.csv`.

Nothing is copied from reports: what was computed is printed.
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
import warnings

warnings.filterwarnings("ignore")

ROOT = work()
D = DEPOSIT
RAWX = ROOT / "FINAL_rebutS" / "02_ДАННЫЕ" / "Фенотипы.xlsx"

MO = {"VK101A": "M1", "VA761A": "M2", "VK934A": "M3"}
TOL = 0.05                       # threshold separating a real discrepancy from rounding
ORDER = ["LI%d" % i for i in range(1, 55)]

hm = en_table(pd.read_csv(INTERMEDIATE / "221_hybrid_means_delivered.csv"))

# ---------------------------------------------------------------- raw data
raw = en_table(pd.read_excel(RAWX, "Гибриды_делянки"))
raw = raw[raw["replicate_type"] == "field_plot"]
# plants -> plot -> year -> mean over two years
pm = raw.groupby(["mother", "line", "year", "replicate", "trait"], as_index=False)["value"].mean()
ym = pm.groupby(["mother", "line", "year", "trait"], as_index=False)["value"].mean()
prim = ym.groupby(["mother", "line", "trait"], as_index=False)["value"].mean()
P = prim.pivot(index=["mother", "line"], columns="trait", values="value")
Y = ym.pivot(index=["mother", "line"], columns=["trait", "year"], values="value")

TRAITS = ["plant_height", "head_diameter", "seed_weight_1000", "oil_content", "seed_yield"]

# ------------------------------------------------- 1. shift hypothesis
print("1. ONE-LINE SHIFT HYPOTHESIS (2022 summary table is offset by +1 position)")
print("   a match is |summary − raw| < %.2f\n" % TOL)
for tr in TRAITS:
    for m in ("M1", "M2", "M3"):
        idx = [l for l in ORDER
               if (m, l) in Y.index and len(hm[(hm["mother"] == m) & (hm["father"] == l)])]
        a22 = np.array([Y.loc[(m, l), (tr, 2022)] for l in idx])
        a23 = np.array([Y.loc[(m, l), (tr, 2023)] for l in idx])
        b = np.array([hm[(hm["mother"] == m) & (hm["father"] == l)][tr].iloc[0] for l in idx])
        n = len(idx)
        ok = np.isfinite(a22) & np.isfinite(a23) & np.isfinite(b)
        m0 = int(sum(abs(b[i] - (a22[i] + a23[i]) / 2) < TOL for i in range(n) if ok[i]))
        m1 = int(sum(abs(b[i] - (a22[i + 1] + a23[i]) / 2) < TOL
                     for i in range(n - 1) if ok[i] and np.isfinite(a22[i + 1])))
        mark = "   <== SHIFT" if m1 > 2 * max(m0, 1) else ""
        print(f"   {tr:15s} {m}: n={n}  no shift={m0:2d}  with shift={m1:2d}{mark}")
    print()

# ------------------------------------------------- 2. compare the two raw sources
print("2. COMPARISON OF PLOT-LEVEL RAW DATA FROM TWO SOURCES")
p1 = en_table(pd.read_parquet(OUT / "221_hybrid_plot_level.parquet"))
p1 = p1.rename(columns={"mother": "mother", "father": "line", "year": "year",
                        "rep": "replicate", "trait": "trait", "value": "value"})
p1["mother"] = p1["mother"].map(MO).fillna(p1["mother"])
p1["trait"] = p1["trait"].replace({"seed_weight_1000": "seed_weight_1000"})
for tr in ("seed_yield", "oil_content", "seed_weight_1000"):
    a = p1[p1["trait"] == tr].set_index(["mother", "line", "year", "replicate"])["value"]
    b = raw[raw["trait"] == tr].set_index(["mother", "line", "year", "replicate"])["value"]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    print(f"   {tr:15s}: shared plots {len(j)}, max |diff| = {(j.a - j.b).abs().max():.6f}")
print()

# ------------------------------------------------- 3. scale of discrepancies
# Each trait is compared to the source that will be used to rebuild it:
# height and diameter — field records (no other source exists for them),
# seed yield, oil content, and 1000-seed weight — `221_hybrid_plot_level.parquet`,
# i.e. the working raw table that already has outliers removed.
Q = (p1.dropna(subset=["value"])
       .groupby(["mother", "line", "year", "trait"], as_index=False)["value"].mean()
       .groupby(["mother", "line", "trait"], as_index=False)["value"].mean()
       .pivot(index=["mother", "line"], columns="trait", values="value"))
SRC = {"plant_height": P, "head_diameter": P,
       "seed_weight_1000": Q, "oil_content": Q, "seed_yield": Q}
NAME = {id(P): "field records", id(Q): "221_hybrid_plot_level.parquet"}

print("3. SCALE OF DISCREPANCIES BETWEEN THE SUMMARY TABLE AND THE RAW DATA")
for tr in TRAITS:
    src = SRC[tr]
    d = np.array([abs(r[tr] - src.loc[(r["mother"], r["father"]), tr])
                  if (r["mother"], r["father"]) in src.index else np.nan
                  for _, r in hm.iterrows()], dtype=float)
    d = d[np.isfinite(d)]
    print(f"   {tr:15s} vs «{NAME[id(src)]:24s}»: cells {len(d)}, "
          f"discordant >{TOL}: {int((d > TOL).sum()):3d}, "
          f"median {np.median(d):.4f}, max {d.max():.4f}")
    if tr in ("plant_height", "head_diameter"):
        # distribution of discrepancies and the share attributable to M1: these
        # quantities support the Methods 2.2 claim that the rebuild captures more
        # than the shift explains (registry row N-21b1)
        d1 = np.array([abs(r[tr] - src.loc[(r["mother"], r["father"]), tr])
                       for _, r in hm[hm["mother"] == "M1"].iterrows()
                       if (r["mother"], r["father"]) in src.index], dtype=float)
        d1 = d1[np.isfinite(d1)]
        for th in (0.5, 2.0):
            print(f"       of which |diff| > {th}: {int((d > th).sum()):3d} "
                  f"(M1 {int((d1 > th).sum())}, outside M1 {int((d > th).sum()) - int((d1 > th).sum())})")

# The rebuild set is stated explicitly, not inferred from a threshold: rebuild
# what is a defect, not an explained processing difference.
#   height, diameter — one-line shift, direction established;
#   1000-seed weight — 11 cells up to 10.35 g, mechanism unknown, no shift signal;
#   oil content — 4 cells up to 3.85 %, and that is outlier removal in the plot-level
#     set that the summary table does not carry: the difference is explained and
#     described in Methods 2.2, so the column is NOT touched;
#   seed yield — no discrepancies beyond 0.0004.
FIX = ["plant_height", "head_diameter", "seed_weight_1000"]
print(f"\n   rebuilt: {FIX}")
print("   oil content is NOT rebuilt: its discrepancy is explained by outlier removal")

# ------------------------------------------------- 4. corrected table
out = hm.copy()
for tr in FIX:
    src = SRC[tr]
    out[tr] = [src.loc[(m, f), tr] if (m, f) in src.index else np.nan
               for m, f in zip(out["mother"], out["father"])]
    print(f"   {tr}: source — {'field records' if src is P else '221_hybrid_plot_level.parquet'}")
out["oil_yield"] = out["seed_yield"] * out["oil_content"] / 100.0
dst = OUT / "221_hybrid_means_corrected.csv"
out.to_csv(dst, index=False, encoding="utf-8-sig")
print(f"\n4. written: {dst}")
