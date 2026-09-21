# -*- coding: utf-8 -*-
"""
Linkage-disequilibrium decay on the analysis panel (54 paternal lines,
merged call set, call rate >= 0.9, MAF >= 0.05 — 81 903 SNPs).

Mean r^2 is computed between markers on the same chromosome by distance bins.
Each chromosome is randomly subsampled to a fixed marker count, otherwise
the pairwise r^2 matrix does not fit in memory; the seed is fixed.
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
import pathlib
import warnings

warnings.filterwarnings("ignore")

D = DEPOSIT
SEED, PER_CHROM = 20260822, 2500
BINS = [(0, 1e5), (1e5, 5e5), (5e5, 1e6), (1e6, 1.5e6), (1.5e6, 2e6),
        (2e6, 3e6), (3e6, 4e6), (4e6, 5e6), (5e6, 1e7)]

z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float)
G[G < 0] = np.nan
samples = [str(x) for x in z["samples"]]
fath = [i for i, n in enumerate(samples) if is_line_id(n)]
g = G[:, fath]
call = 1 - np.isnan(g).mean(1)
p = np.nanmean(g, 1) / 2
maf = np.minimum(p, 1 - p)
keep = (call >= 0.9) & (maf >= 0.05)
g = g[keep]
pos = z["pos"][keep].astype(float)
chrom = np.array([str(c) for c in z["chrom"]])[keep]
g = np.where(np.isnan(g), np.nanmean(g, 1, keepdims=True), g)
print(f"panel: {g.shape[0]} markers x {g.shape[1]} lines")

acc = {b: [0.0, 0] for b in BINS}
rng = np.random.default_rng(SEED)
for c in np.unique(chrom):
    m = np.where(chrom == c)[0]
    if len(m) < 200:
        continue
    take = np.sort(rng.choice(m, size=min(PER_CHROM, len(m)), replace=False))
    X, pp = g[take], pos[take]
    Xc = X - X.mean(1, keepdims=True)
    sd = Xc.std(1)
    ok = sd > 1e-9
    Xc, pp, sd = Xc[ok], pp[ok], sd[ok]
    Z = Xc / (sd[:, None] * np.sqrt(Xc.shape[1]))
    R2 = (Z @ Z.T) ** 2
    Dm = np.abs(pp[:, None] - pp[None, :])
    iu = np.triu_indices(len(pp), 1)
    dd, rr = Dm[iu], R2[iu]
    for b in BINS:
        sel = (dd >= b[0]) & (dd < b[1])
        if sel.sum():
            acc[b][0] += float(rr[sel].sum())
            acc[b][1] += int(sel.sum())

print(f"{'distance, bp':>26s}   {'mean r²':>10s}   {'pairs':>12s}")
for b in BINS:
    ssum, n = acc[b]
    if n:
        print(f"{b[0]:>11,.0f}–{b[1]:>12,.0f}   {ssum / n:10.4f}   {n:12,}")
