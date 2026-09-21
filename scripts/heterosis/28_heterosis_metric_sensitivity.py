# -*- coding: utf-8 -*-
"""
Sensitivity of the “distance ↔ yield” result (Results 3.3) to the choice of distance metric
and to the denominator (159 versus 162 hybrids). Four combinations:
  1. old metric (binary indicator “both parents are opposite homozygotes”, var2)
     on the former data (summary table 221_hybrid_means_delivered.csv, n=159)
  2. the same metric on the corrected data (plot-level raw 221_hybrid_plot_level.parquet, n=162)
  3. the paper’s continuous metric, mean(|round(gm)-round(gf)|/2) (merged VCF), on the former data
  4. the same metric on the corrected data — what is printed in Results 3.3 / Methods 2.8

Metric 1 was never a method of the paper: it appears only in the internal, later
discarded check verify_all_results.py.OLD (see brain\\STRONGEST_RESULT_AUDIT.md, K1). Metric 3 is
the paper’s method from the original setup (Phase 9a, 2026-06-02) unchanged.

Output: 28_heterosis_metric_sensitivity.csv, in Supplementary S4 order.
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
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")

REPO = DEPOSIT
DATA = DEPOSIT
EXT_ROOT = work()
VAR2_VCF = EXT_ROOT / "6_новые данные_ответы + исходники +vcf" / "sunflower_var2_vcf_merged.vcf"
OUT_CSV = OUT / "28_heterosis_metric_sensitivity.csv"

SEQ = {i: f"LI{i}" for i in range(1, 33)}
SEQ.update({33: "LI34", 34: "LI35"}); SEQ.update({i: f"LI{i+2}" for i in range(35, 53)})
SEQ.update({53: "LI33", 54: "LI36", 55: "VA761", 56: "VK101", 57: "VK934"})
MOTH = {"M1": "VK101", "M2": "VA761", "M3": "VK934"}
LONG = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}

# ---------- data (phenotypes) ----------
hm = en_table(pd.read_csv(OUT / "221_hybrid_means_delivered.csv", encoding="utf-8-sig"))  # "former", n=159
pl = en_table(pd.read_parquet(OUT / "221_hybrid_plot_level.parquet"))  # plot-level raw, 162 combinations
fixed = (pl[pl["trait"] == "seed_yield"].groupby(["mother", "father"])["value"].mean()
         .reset_index().rename(columns={"value": "seed_yield"}))
assert len(hm) == 159 and len(fixed) == 162

# ---------- old metric: binary indicator on var2 ----------
print("Loading var2 VCF (several minutes)...", flush=True)
import allel
cs = allel.read_vcf(str(VAR2_VCF), fields=["variants/CHROM", "variants/POS", "calldata/GT", "samples"])
GT_old = allel.GenotypeArray(cs["calldata/GT"]).to_n_alt().astype(float)  # fill=0 by default, as in .OLD
GT_old[GT_old < 0] = np.nan
lines_old = np.array([SEQ.get(int(float(s)), str(s)) for s in cs["samples"]])
G_all_old = np.where(np.isnan(GT_old), np.nanmean(GT_old, 1, keepdims=True), GT_old)
idx_old = {n: i for i, n in enumerate(lines_old)}
print(f"var2: {GT_old.shape[0]} SNP x {GT_old.shape[1]} samples", flush=True)
del cs

def old_metric(mother_vcf_name, father):
    gm = G_all_old[:, idx_old[mother_vcf_name]]
    gf = G_all_old[:, idx_old[father]]
    return float(np.mean(np.abs(np.round(gm) - np.round(gf)) == 2))

# ---------- new metric: continuous on the merged VCF (Methods 2.8) ----------
zg = np.load(DATA / "23_merged_genotypes.npz", allow_pickle=True)
G = zg["genotypes"].astype(float); G[G < 0] = np.nan
samples_m = [str(x) for x in zg["samples"]]; chrom_m = [str(c) for c in zg["chrom"]]
CHRSET = set(["CM0078%02d.2" % n for n in range(90, 100)] + ["CM0079%02d.2" % n for n in range(0, 7)])
on_chr = np.array([c in CHRSET for c in chrom_m])
call_rate = np.mean(~np.isnan(G), axis=1); af = np.nanmean(G, axis=1) / 2; maf = np.minimum(af, 1 - af)
keep = on_chr & (call_rate >= 0.5) & (maf >= 0.05)
X = G[keep]; X = np.where(np.isnan(X), np.nanmean(X, axis=1, keepdims=True), X)
col_m = {s: i for i, s in enumerate(samples_m)}
assert X.shape[0] == 161791

def new_metric(mother_vcf_name, father):
    return float(np.mean(np.abs(np.round(X[:, col_m[mother_vcf_name]]) - np.round(X[:, col_m[father]])) / 2))

def pearson_ci(r, n, alpha=0.05):
    z = np.arctanh(r); se = 1 / np.sqrt(n - 3); zc = stats.norm.ppf(1 - alpha / 2)
    return float(np.tanh(z - zc * se)), float(np.tanh(z + zc * se))

def run(label, metric_fn, df, mother_col, mother_map, father_col, yield_col):
    dist = [metric_fn(mother_map[m], f) for m, f in zip(df[mother_col], df[father_col])]
    d = df.assign(dist=dist)
    n = len(d)
    r, p = stats.pearsonr(d["dist"], d[yield_col])
    lo, hi = pearson_ci(r, n)
    row = {"metric": label[0], "data": label[1], "n": n, "r": round(r, 4), "p": p,
           "p_rounded": f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}",
           "CI95_lo": round(lo, 4), "CI95_hi": round(hi, 4)}
    print(row, flush=True)
    return row

rows = [
    run(("binary_var2", "previous_n159"), old_metric, hm, "mother", MOTH, "father", "seed_yield"),
    run(("binary_var2", "corrected_n162"), old_metric, fixed, "mother", LONG, "father", "seed_yield"),
    run(("continuous_methods_2_8", "previous_n159"), new_metric, hm, "mother", MOTH, "father", "seed_yield"),
    run(("continuous_methods_2_8", "corrected_n162, = Results 3.3"), new_metric, fixed,
        "mother", LONG, "father", "seed_yield"),
]
pd.DataFrame(rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUT_CSV}")
