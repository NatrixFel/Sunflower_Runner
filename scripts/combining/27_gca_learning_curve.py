# -*- coding: utf-8 -*-
"""Z4. Evaluate the proposal to pivot toward local genomic prediction.
Computes: effective n, paternal GCA prediction accuracy under leave-one-father-out,
the learning curve (accuracy as a function of the number of training fathers).
Does not modify anything.
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
import numpy as np, pandas as pd, pathlib
from scipy.optimize import minimize_scalar
import warnings; warnings.filterwarnings("ignore")

D = DEPOSIT
plot = en_table(pd.read_parquet(OUT / "22_hybrid_plot_level.parquet"))
blup = en_table(pd.read_csv(OUT / "22_hybrid_blup_phenotypes.csv"))
blup[["mother", "father"]] = blup["hybrid"].str.split("_", n=1, expand=True)
TR = ["seed_yield", "oil_content", "seed_weight_1000", "oil_yield"]
CONF = ["LI29", "LI30"]

print("=" * 86); print("A. EFFECTIVE n"); print("=" * 86)
print(f"  hybrids (observations at the combination level): {blup['hybrid'].nunique()}")
print(f"  paternal lines (independent genotypes):          {blup['father'].nunique()}")
print(f"  maternal testers:                                {blup['mother'].nunique()}")
print(f"  testcrosses per father:                          {blup.groupby('father').size().unique()}")
pl = plot.groupby(["father"]).size()
print(f"  plots per father (over all traits):              median {pl.median():.0f}, min {pl.min()}, max {pl.max()}")
print("  rank of the genotypic part of the design for paternal effects = 54;")
print("  162 hybrids give 54 independent paternal units, each measured 3 times (3 testers).")

# ---------- genomic relationship matrix among fathers ----------
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
def cn(c):
    try: return int(str(c).replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1
chrom = np.array([cn(c) for c in z["chrom"]]); samp = en_ids(z["samples"].astype(str))
cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (maf >= 0.05)
M = G[keep]
M = np.where(np.isnan(M), np.nanmean(M, 1, keepdims=True), M)
fathers = sorted(blup["father"].unique())
idx = {s: i for i, s in enumerate(samp)}
Mf = M[:, [idx[f] for f in fathers]]
p = Mf.mean(1) / 2
Zc = Mf - 2 * p[:, None]
K = (Zc.T @ Zc) / (2 * np.sum(p * (1 - p)))
K += np.eye(len(fathers)) * 1e-6
print(f"\n  panel for K: {Mf.shape[0]} SNP, {len(fathers)} fathers")
print(f"  mean off-diagonal relatedness: {K[np.triu_indices(len(fathers),1)].mean():.4f}, "
      f"diagonal {np.diag(K).mean():.4f}")

# ---------- GBLUP on paternal means ----------
def gblup_cv(y, K, train_idx, test_idx):
    """Train on train, predict test. REML on the variance ratio using the training part."""
    ytr = y[train_idx]; Ktr = K[np.ix_(train_idx, train_idx)]
    ytr = ytr - ytr.mean()
    n = len(ytr)
    def negll(loglam):
        lam = np.exp(loglam)
        V = Ktr + np.eye(n) / lam
        s, ld = np.linalg.slogdet(V)
        a = np.linalg.solve(V, ytr)
        return 0.5 * (ld + n * np.log(ytr @ a))
    r = minimize_scalar(negll, bounds=(-8, 8), method="bounded")
    lam = np.exp(r.x)
    V = Ktr + np.eye(n) / lam
    alpha = np.linalg.solve(V, ytr)
    return K[np.ix_(test_idx, train_idx)] @ alpha

rng = np.random.default_rng(20260818)
print("\n" + "=" * 86)
print("B. PATERNAL GCA PREDICTION: leave-one-father-out on means over three testers")
print("=" * 86)
res_loo = {}
for tr in TR:
    sub = blup.dropna(subset=[tr])
    fm = sub.groupby("father")[tr].mean()
    fl = [f for f in fathers if f in fm.index]
    if tr in ("oil_content", "seed_weight_1000"):
        fl = [f for f in fl if f not in CONF]
    y = fm.reindex(fl).to_numpy(float)
    ii = [fathers.index(f) for f in fl]
    Ksub = K[np.ix_(ii, ii)]
    pred = np.empty(len(fl))
    for k in range(len(fl)):
        tr_i = [j for j in range(len(fl)) if j != k]
        pred[k] = gblup_cv(y, Ksub, tr_i, [k])[0]
    r = np.corrcoef(pred, y)[0, 1]
    res_loo[tr] = (len(fl), r)
    print(f"  {tr:14s} fathers {len(fl):3d}   r(predicted, observed) = {r:+.3f}")

print("\n" + "=" * 86)
print("C. LEARNING CURVE: accuracy as a function of the number of training fathers")
print("=" * 86)
SIZES = [10, 20, 30, 40, 45, 50]
NREP = 40
print(f"  {'trait':14s} " + "  ".join(f"n={s:<3d}" for s in SIZES) + "   LOO")
for tr in TR:
    sub = blup.dropna(subset=[tr])
    fm = sub.groupby("father")[tr].mean()
    fl = [f for f in fathers if f in fm.index]
    if tr in ("oil_content", "seed_weight_1000"):
        fl = [f for f in fl if f not in CONF]
    y = fm.reindex(fl).to_numpy(float)
    ii = [fathers.index(f) for f in fl]
    Ksub = K[np.ix_(ii, ii)]
    line = []
    for s in SIZES:
        rs = []
        for _ in range(NREP):
            perm = rng.permutation(len(fl))
            trn, tst = perm[:s], perm[s:]
            if len(tst) < 5: continue
            pr = gblup_cv(y, Ksub, list(trn), list(tst))
            if np.std(pr) < 1e-12: continue
            rs.append(np.corrcoef(pr, y[tst])[0, 1])
        line.append(np.mean(rs) if rs else np.nan)
    print(f"  {tr:14s} " + "  ".join(f"{v:+.3f}" for v in line) + f"   {res_loo[tr][1]:+.3f}")

print("\n" + "=" * 86)
print("D. RESCALE OUR NUMBERS TO THE SORGHUM PAPER 'ACCURACY' SCALE (r / sqrt(H²))")
print("=" * 86)
acc = en_table(pd.read_csv(OUT / "27_gs_accuracy_blup.csv", comment="#"))
for _, row in acc.iterrows():
    H2 = row["H2_of_combination_mean"]
    if pd.notna(H2):
        print(f"  {row['trait']:14s} predictive ability {row['GS_accuracy_CV1']:.3f} | "
              f"H² {H2:.3f} | accuracy = r/sqrt(H²) = {row['GS_accuracy_CV1']/np.sqrt(H2):.3f}")
    else:
        print(f"  {row['trait']:14s} predictive ability {row['GS_accuracy_CV1']:.3f} | H² not estimated")
print("\n  Sorghum (Maulana et al. 2023): H²(GY) = 0.23; reported accuracy GY = 0.58 (additive) / 0.67 (full)")
print(f"  -> their predictive ability for seed yield = 0.58*sqrt(0.23) = {0.58*np.sqrt(0.23):.3f} "
      f"(full model {0.67*np.sqrt(0.23):.3f})")
