# -*- coding: utf-8 -*-
"""
Reproduce the second group’s recompute at n = 162.
Nothing is copied from the prompt: we compute from the plot-level raw data and merged ourselves.
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
from scipy import stats
import pathlib

D = DEPOSIT
# ---------- 1. hybrid means from plot-level raw data (162) ----------
plot = en_table(pd.read_parquet(OUT / "22_hybrid_plot_level.parquet"))
means162 = (plot.groupby(["mother", "father", "trait"])["value"].mean()
            .unstack("trait").reset_index())
print("hybrid means from plot-level data:", len(means162))

# ---------- 2. phase3c summary table (159) ----------
hm = en_table(pd.read_csv(OUT / "22_hybrid_means_delivered.csv"))
key159 = set(zip(hm["mother_long"], hm["father"]))
key162 = set(zip(means162["mother"], means162["father"]))
print("in plot-level data but not in the summary:", sorted(key162 - key159))

# ---------- 3. panel and genetic distance ----------
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
def cn(c):
    try: return int(str(c).replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1
chrom = np.array([cn(c) for c in z["chrom"]]); samp = en_ids(z["samples"].astype(str))
cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.5) & (maf >= 0.05)
Gk = G[keep]; Gk = np.where(np.isnan(Gk), np.nanmean(Gk, 1, keepdims=True), Gk)
print("distance panel:", Gk.shape[0], "SNP")
MO = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
idx = {s: i for i, s in enumerate(samp)}
def dist(mother_long, father):
    gm = np.round(Gk[:, idx[MO[mother_long]]]); gf = np.round(Gk[:, idx[father]])
    return float(np.mean(np.abs(gm - gf) / 2))

means162["dist"] = [dist(m, f) for m, f in zip(means162["mother"], means162["father"])]

# ---------- 4. distance–seed-yield correlations ----------
def corrs(df, label):
    rows = []
    x, y = df["dist"].values, df["seed_yield"].values
    m = ~np.isnan(y)
    r, p = stats.pearsonr(x[m], y[m])
    rows.append(("ALL", int(m.sum()), r, p))
    for mom in ["VK934A", "VK101A", "VA761A"]:
        s = df[df["mother"] == mom]
        xx, yy = s["dist"].values, s["seed_yield"].values
        mm = ~np.isnan(yy)
        rr, pp = stats.pearsonr(xx[mm], yy[mm])
        rows.append((mom, int(mm.sum()), rr, pp))
    print(f"\n--- distance x seed yield, {label} ---")
    for g, n, r, p in rows:
        print(f"  {g:8s} n={n:3d}  r={r:.4f}  p={p:.5f}   (rounded r={r:.3f}, p={p:.4f})")
    return rows

res162 = corrs(means162, "n = 162 (plot-level means)")
sub159 = means162[[ (m, f) in key159 for m, f in zip(means162["mother"], means162["father"]) ]]
res159 = corrs(sub159, "n = 159 (summary-table combinations only)")

# for reconciliation with the old figure: on the summary-table values themselves
hm2 = hm.copy()
hm2["dist"] = [dist(m, f) for m, f in zip(hm2["mother_long"], hm2["father"])]
hm2 = hm2.drop(columns=["mother"]).rename(columns={"mother_long": "mother"})
res_hm = corrs(hm2, "n = 159, values from the phase3c summary table")

# ---------- 5. Baker ratio ----------
def baker(df, trait):
    tab = df.pivot_table(index="father", columns="mother", values=trait)
    g = np.nanmean(tab.values)
    gf = np.nanmean(tab.values, 1) - g
    gm = np.nanmean(tab.values, 0) - g
    sca = tab.values - g - gf[:, None] - gm[None, :]
    vg = np.nanvar(np.concatenate([gf, gm]))
    vs = np.nanvar(sca.ravel())
    return 2 * vg / (2 * vg + vs)

print("\n--- Baker ratio ---")
TR = [("seed_yield", "seed_yield"), ("oil_content", "oil_content"), ("seed_weight_1000", "seed_weight_1000")]
for name, hmcol in TR:
    b162 = baker(means162, name)
    b159 = baker(sub159, name)
    bhm = baker(hm2, hmcol)
    print(f"  {name:12s} 162={b162:.4f} ({b162:.2f})  159-subset={b159:.4f} ({b159:.2f})  "
          f"summary-159={bhm:.4f} ({bhm:.2f})")

# ---------- 6. which source the maternal GCA and paternal index were computed from ----------
print("\n--- GCA source ---")
gcaf = en_table(pd.read_csv(OUT / "25_gca_fathers.csv")).set_index("father")
gcam = en_table(pd.read_csv(OUT / "25_gca_mothers.csv"))
def gca_from(df, trait, col_out):
    tab = df.pivot_table(index="father", columns="mother", values=trait)
    g = np.nanmean(tab.values)
    return (pd.Series(np.nanmean(tab.values, 1) - g, index=tab.index),
            pd.Series(np.nanmean(tab.values, 0) - g, index=tab.columns))
for name, hmcol in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"), ("seed_weight_1000", "seed_weight_1000")]:
    f162, m162 = gca_from(means162, name, hmcol)
    f159, m159 = gca_from(sub159, name, hmcol)
    fhm, mhm = gca_from(hm2, hmcol, hmcol)
    stored_f = gcaf[hmcol]
    print(f"  {hmcol:16s} corr(stored, 162)={np.corrcoef(stored_f.reindex(f162.index), f162)[0,1]:.6f} "
          f"| corr(stored, summary-159)={np.corrcoef(stored_f.reindex(fhm.index), fhm)[0,1]:.6f} "
          f"| max|diff| vs 162 = {np.max(np.abs(stored_f.reindex(f162.index).values - f162.values)):.4f} "
          f"| vs summary = {np.max(np.abs(stored_f.reindex(fhm.index).values - fhm.values)):.4f}")
print("\n  maternal GCA, stored:")
print(gcam[["mother_long", "seed_yield", "oil_content", "seed_weight_1000", "plant_height"]].to_string(index=False))
for name, hmcol in [("seed_yield", "seed_yield"), ("oil_content", "oil_content"), ("seed_weight_1000", "seed_weight_1000")]:
    _, m162 = gca_from(means162, name, hmcol)
    _, mhm = gca_from(hm2, hmcol, hmcol)
    print(f"  {hmcol:16s} 162: " + ", ".join(f"{k}={v:+.4f}" for k, v in m162.items()) +
          " | summary-159: " + ", ".join(f"{k}={v:+.4f}" for k, v in mhm.items()))
