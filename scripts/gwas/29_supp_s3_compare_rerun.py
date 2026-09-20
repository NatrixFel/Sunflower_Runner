# -*- coding: utf-8 -*-
"""F3 / S3. Re-run of the comparison “our Python BLINK vs GAPIT-BLINK on the same input”.

Supplementary claim: “reproduced the GAPIT-BLINK top-1 SNP for 5 of 9 traits and the
same chromosome for a sixth”. It relied on `compare_no_sugar.csv`, computed before
the recalculation. Here the same comparison is recomputed from the source files of
the separate implementation, with the match counted explicitly rather than by eye.
The GAPIT run used for the check: `29_gapit_blink.R` (BLINK, VanRaden, PCA.total = 5).

Input is read-only, from the external directory of the separate implementation.
Output is written to `brain\\`, not next to the sources.
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
import sys, pathlib, warnings
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")
ROOT = DEPOSIT
SCR = MODELS
use_models()
from blink import blink  # noqa: E402

EXT = work()
NS = EXT / "7_ручной обсчет воронежская" / "no_sugar" / "no_sugar"
GENO = NS / "gapit" / "input" / "gapit_genotypes.txt"
PHENO = NS / "pheno" / "gapit_phenotypes.csv"
BL = NS / "gapit" / "output" / "blink"

TRAITS = ["oil_content_pct", "seed_weight_1000_g", "height_cm_mean", "head_diameter_cm_mean",
          "hull_pct", "emergence_to_flowering_days", "central_to_side_flowering_days",
          "self_fertility_self_pollination_mean", "self_fertility_open_pollination_mean"]

# --- their genotypes
g = en_table(pd.read_csv(GENO, sep="\t", header=0, dtype=str))
ids = g["rs"].values
gchr = np.array([int(c) for c in g["chrom"]])
gpos = g["pos"].astype(int).values
samp = g.columns[11:]
M = g.iloc[:, 11:].apply(pd.to_numeric, errors="coerce").values.astype(float)
print("separate-implementation input: %d SNP × %d lines" % M.shape)

# --- their phenotypes and principal components (PCA.total = 5, as in their run)
Y = en_table(pd.read_csv(PHENO, dtype={"Taxa": str}))
Y["Taxa"] = Y["Taxa"].apply(lambda s: "%.1f" % float(s))
Y = Y.set_index("Taxa").reindex(list(samp))
Mi = np.where(np.isnan(M), np.nanmean(M, axis=1, keepdims=True), M)
Gc = (Mi - Mi.mean(axis=1, keepdims=True)).T
Gc -= Gc.mean(axis=0, keepdims=True)
U, S, _ = np.linalg.svd(Gc, full_matrices=False)
PC5 = U[:, :5] * S[:5]

def their_top(trait):
    f = list((BL / trait).glob("GAPIT.Association.GWAS_Results.BLINK.*.csv"))[0]
    r = en_table(pd.read_csv(f))
    t = r.loc[r["P.value"].idxmin()]
    return int(t.Chr), int(t.Pos), float(t["P.value"])

rows = []
for tr in TRAITS:
    tchr, tpos, tp = their_top(tr)
    y = Y[tr].values.astype(float)
    res = blink(y, Mi, gchr, gpos, PCs=PC5)
    p = np.where(np.isnan(res["pvals"]), np.inf, res["pvals"])
    i = int(np.argmin(p))
    ochr, opos, op = int(gchr[i]), int(gpos[i]), float(res["pvals"][i])
    same_snp = (ochr == tchr) and (opos == tpos)
    same_chr = (ochr == tchr)
    within1mb = same_chr and abs(opos - tpos) < 1_000_000
    rows.append(dict(trait=tr,
                     gapit_chr=tchr, gapit_pos=tpos, gapit_p=tp,
                     ours_chr=ochr, ours_pos=opos, ours_p=op,
                     same_snp=same_snp, same_chr=same_chr, within_1mb=within1mb))
    print("  %-40s GAPIT chr%d:%d (p=%.1e)  ours chr%d:%d (p=%.1e)  %s"
          % (tr, tchr, tpos, tp, ochr, opos, op,
             "same SNP" if same_snp else ("same chromosome" if same_chr else "diverge")))

df = pd.DataFrame(rows)
n_snp = int(df["same_snp"].sum())
n_chr_only = int((df["same_chr"] & ~df["same_snp"]).sum())
print("\nTOTAL: same top-1 SNP — %d of %d traits; same chromosome without SNP match — %d; "
      "within ±1 Mb — %d" % (n_snp, len(df), n_chr_only, int(df["within_1mb"].sum())))
out = ROOT / "brain" / "29_supp_s3_compare_rerun.csv"
df.to_csv(out, index=False, encoding="utf-8-sig")
print("written:", out)
