# -*- coding: utf-8 -*-
"""D2. Association of call rate with principal-component coordinates.
Two panels are compared: var2 with missing calls substituted by the reference homozygote
(historical, defect D-30) and the merged set with missing calls stored correctly."""

import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import numpy as np, pandas as pd, pathlib, pickle
from scipy import stats
SCR = OUT
D = DEPOSIT
SEQ = {n: f"LI{n}" for n in range(1, 33)}
SEQ[33] = "LI34"; SEQ[34] = "LI35"
SEQ.update({n: f"LI{n+2}" for n in range(35, 53)})
SEQ[53] = "LI33"; SEQ[54] = "LI36"; SEQ[55] = "VA761"; SEQ[56] = "VK101"; SEQ[57] = "VK934"
CONF = {"LI29", "LI30"}

M2 = np.load(SCR / "var2_M.npy")
s1, k1, s2, k2 = pickle.load(open(SCR / "vcf_keys.pkl", "rb"))
idx2 = {s: i for i, s in enumerate(s2)}
fathers = [SEQ[n] for n in range(1, 55)]
cols54 = [idx2[str(n)] for n in range(1, 55)]
raw54 = M2[:, cols54]
call_rate_var2 = np.mean(raw54 >= 0, 0)

# ---- var2 panel WITH SUBSTITUTION: missing becomes 0 (reference homozygote) ----
sub54 = np.where(raw54 < 0, 0, raw54).astype(float)
sub54[sub54 == 3] = 2
af = sub54.mean(1) / 2; maf = np.minimum(af, 1 - af)
mask_sub = maf >= 0.05
print(f"var2 with substitution, MAF >= 0.05 and missing-call filter (inert): {int(mask_sub.sum())} markers")
print(f"   for cross-check: the withdrawn manuscript edition had 263 498 (54 lines)")

# ---- merged set, missing calls stored correctly ----
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
samp = en_ids(z["samples"].astype(str))
chrom = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 if str(c).startswith("CM") else -1
                  for c in z["chrom"]])
i54 = [samp.index(f) for f in fathers]
G54 = G[:, i54]
cr = np.mean(~np.isnan(G54), 1); afm = np.nanmean(G54, 1) / 2; mafm = np.minimum(afm, 1 - afm)
mask_m = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (mafm >= 0.05)
print(f"merged set, analysis panel: {int(mask_m.sum())} markers (81 903 in the paper)")
Gm = G54[mask_m]
call_rate_merged = np.mean(~np.isnan(G54), 0)
Gm_imp = np.where(np.isnan(Gm), np.nanmean(Gm, 1, keepdims=True), Gm)

def pca(X_markers_by_samples, standardise):
    X = X_markers_by_samples.T.astype(float)
    X = X - X.mean(0, keepdims=True)
    if standardise:
        sd = X.std(0, ddof=0); sd[sd < 1e-12] = 1.0
        X = X / sd
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U * S, S ** 2 / np.sum(S ** 2)

conf_ind = np.array([1.0 if f in CONF else 0.0 for f in fathers])

print("\n" + "=" * 100)
print("D2. CORRELATION OF SAMPLE CALL RATE WITH PRINCIPAL-COMPONENT COORDINATES")
print("=" * 100)
res = []
for panel, X, cr_vec in (("var2_with_substitution", sub54[mask_sub], call_rate_var2),
                         ("merged_missing_ok", Gm_imp, call_rate_merged)):
    for std, sname in ((True, "standardised"), (False, "centred_only")):
        PC, var = pca(X, std)
        line = f"\n{panel}, {sname}: PC1 {var[0]*100:.2f} %, PC2 {var[1]*100:.2f} %, PC3 {var[2]*100:.2f} %"
        print(line)
        for k in range(3):
            r, p = stats.pearsonr(cr_vec, PC[:, k])
            rs, ps = stats.spearmanr(cr_vec, PC[:, k])
            print(f"    call rate x PC{k+1}: r = {r:+.3f} (p = {p:.2e}) | Spearman {rs:+.3f}")
            res.append(dict(panel=panel, preprocess=sname, component=k+1, r=r, p=p))
        rc = abs(np.corrcoef(PC[:, 0], conf_ind)[0, 1])
        print(f"    PC1 x use type: |r| = {rc:.3f}")
        # confectionery position on PC1 in SD units
        zc = (PC[:, 0] - PC[:, 0].mean()) / PC[:, 0].std()
        print(f"    LI29 on PC1: {zc[fathers.index('LI29')]:+.2f} SD | "
              f"LI30: {zc[fathers.index('LI30')]:+.2f} SD")
pd.DataFrame(res).to_csv(SCR / "coverage_pca.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 100)
print("CONTROL: call rate versus use type (no PCA at all)")
print("=" * 100)
for nm, v in (("var2", call_rate_var2), ("merged", call_rate_merged)):
    r, p = stats.pearsonr(conf_ind, v)
    print(f"  {nm:14s} correlation of confectionery-type indicator with call rate: r = {r:+.3f}, p = {p:.3f}")
