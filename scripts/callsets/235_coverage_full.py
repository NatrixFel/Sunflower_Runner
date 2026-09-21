# -*- coding: utf-8 -*-
"""D2 (partial correlations) and D3 (markers with complete coverage).
On a set where every marker is called in all 54 fathers, substitution is impossible by construction."""

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
SEQ[53] = "LI33"; SEQ[54] = "LI36"
CONF = {"LI29", "LI30"}
fathers = [SEQ[n] for n in range(1, 55)]
conf_ind = np.array([1.0 if f in CONF else 0.0 for f in fathers])

z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
samp = en_ids(z["samples"].astype(str))
chrom = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 if str(c).startswith("CM") else -1
                  for c in z["chrom"]])
G54 = G[:, [samp.index(f) for f in fathers]]
cr_line = np.mean(~np.isnan(G54), 0)

M2 = np.load(SCR / "var2_M.npy")
s1, k1, s2, k2 = pickle.load(open(SCR / "vcf_keys.pkl", "rb"))
idx2 = {s: i for i, s in enumerate(s2)}
raw54 = M2[:, [idx2[str(n)] for n in range(1, 55)]]
cr_var2 = np.mean(raw54 >= 0, 0)

def pca(X, standardise=True):
    A = X.T.astype(float); A = A - A.mean(0, keepdims=True)
    if standardise:
        sd = A.std(0, ddof=0); sd[sd < 1e-12] = 1.0; A = A / sd
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U * S, S ** 2 / np.sum(S ** 2)

def partial(x, y, z_):
    """partial correlation of x,y given z"""
    rxy = np.corrcoef(x, y)[0, 1]; rxz = np.corrcoef(x, z_)[0, 1]; ryz = np.corrcoef(y, z_)[0, 1]
    return (rxy - rxz * ryz) / np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))

print("=" * 100)
print("D2, supplement. PARTIAL CORRELATION OF PC1 WITH USE TYPE GIVEN CALL RATE")
print("=" * 100)
sub = np.where(raw54 < 0, 0, raw54).astype(float); sub[sub == 3] = 2
af = sub.mean(1) / 2; maf = np.minimum(af, 1 - af)
PCs, _ = pca(sub[maf >= 0.05], True)
r_full = abs(np.corrcoef(PCs[:, 0], conf_ind)[0, 1])
r_part = abs(partial(PCs[:, 0], conf_ind, cr_var2))
print(f"  var2 with substitution:  |r| = {r_full:.3f} -> partial |r| = {r_part:.3f} "
      f"(call rate explains {100*(1-r_part/r_full):.0f} % of the association)")

crm = np.mean(~np.isnan(G54), 1); afm = np.nanmean(G54, 1) / 2; mafm = np.minimum(afm, 1 - afm)
mask_m = (chrom >= 1) & (chrom <= 17) & (crm >= 0.9) & (mafm >= 0.05)
Gm = G54[mask_m]; Gi = np.where(np.isnan(Gm), np.nanmean(Gm, 1, keepdims=True), Gm)
PCm, _ = pca(Gi, True)
r_fullm = abs(np.corrcoef(PCm[:, 0], conf_ind)[0, 1])
r_partm = abs(partial(PCm[:, 0], conf_ind, cr_line))
print(f"  merged panel:            |r| = {r_fullm:.3f} -> partial |r| = {r_partm:.3f} "
      f"(call rate explains {100*(1-r_partm/r_fullm):.0f} % of the association)")

print("\n" + "=" * 100)
print("D3. MARKERS CALLED IN ALL 54 FATHERS WITHOUT EXCEPTION")
print("=" * 100)
full_cov = np.all(~np.isnan(G54), 1) & (chrom >= 1) & (chrom <= 17)
print(f"  markers with complete coverage (call rate = 1.000): {int(full_cov.sum())}")
for thr in (0.05, 0.10):
    m = full_cov & (mafm >= thr)
    print(f"     of them MAF >= {thr}: {int(m.sum())}")
mask_fc = full_cov & (mafm >= 0.05)
n_fc = int(mask_fc.sum())
if n_fc < 200:
    print("  WARNING: the set is small; the calculation is reported with a caveat and cannot replace the main panel")
Gfc = G54[mask_fc]
print(f"  check: missing cells in the set {int(np.isnan(Gfc).sum())} (must be 0)")

for std, nm in ((True, "standardised"), (False, "centred_only")):
    P, var = pca(Gfc, std)
    zc = (P[:, 0] - P[:, 0].mean()) / P[:, 0].std()
    r = abs(np.corrcoef(P[:, 0], conf_ind)[0, 1])
    print(f"\n  PCA on complete coverage, {nm}: PC1 {var[0]*100:.2f} %, PC2 {var[1]*100:.2f} %")
    print(f"    PC1 x use type: |r| = {r:.3f}")
    print(f"    LI29 on PC1: {zc[fathers.index('LI29')]:+.2f} SD | LI30: {zc[fathers.index('LI30')]:+.2f} SD")
    order = np.argsort(-np.abs(zc))[:5]
    print(f"    five most distant lines by |PC1|: " +
          ", ".join(f"{fathers[i]} ({zc[i]:+.2f})" for i in order))
    r_p = abs(partial(P[:, 0], conf_ind, cr_line))
    print(f"    partial |r| given call rate: {r_p:.3f}")

print("\n" + "=" * 100)
print("D3, control without PCA: genetic distance of confectionery to oilseed lines on complete coverage")
print("=" * 100)
X = Gfc.T
def dist_matrix(A):
    n = A.shape[0]; Dm = np.zeros((n, n))
    for i in range(n):
        Dm[i] = np.mean(np.abs(A - A[i]), 1) / 2
    return Dm
Dm = dist_matrix(X)
iu = np.triu_indices(54, 1)
oil_idx = [i for i, f in enumerate(fathers) if f not in CONF]
c_idx = [fathers.index(f) for f in ("LI29", "LI30")]
d_oil_oil = np.mean([Dm[i, j] for i in oil_idx for j in oil_idx if i < j])
d_conf_oil = np.mean([Dm[c, o] for c in c_idx for o in oil_idx])
d_conf_conf = Dm[c_idx[0], c_idx[1]]
print(f"  mean oilseed-oilseed distance:           {d_oil_oil:.4f}")
print(f"  mean confectionery-oilseed distance:     {d_conf_oil:.4f}  "
      f"(ratio {d_conf_oil/d_oil_oil:.3f})")
print(f"  LI29-LI30 distance between themselves:   {d_conf_conf:.4f}")
# same on the main panel for comparison
Xm = Gi.T; Dm2 = dist_matrix(Xm)
d2_oo = np.mean([Dm2[i, j] for i in oil_idx for j in oil_idx if i < j])
d2_co = np.mean([Dm2[c, o] for c in c_idx for o in oil_idx])
print(f"  for comparison, main panel 81 903: oilseed {d2_oo:.4f}, "
      f"confectionery-oilseed {d2_co:.4f} (ratio {d2_co/d2_oo:.3f}), "
      f"LI29-LI30 {Dm2[c_idx[0], c_idx[1]]:.4f}")

# how distinctive the pair is: rank of each line’s mean distance to the others
mean_d = np.array([np.mean(np.delete(Dm[i], i)) for i in range(54)])
rank = np.argsort(-mean_d)
print("\n  mean distance of a line to all others (complete coverage), five largest:")
for i in rank[:5]:
    print(f"    {fathers[i]:8s} {mean_d[i]:.4f}" + ("   <- confectionery" if fathers[i] in CONF else ""))
