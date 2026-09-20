# -*- coding: utf-8 -*-
"""A4. Downward bias of residual-heterozygosity estimates under the merge rule
“homozygote takes priority over heterozygote”.

Counts the fraction of heterozygous calls in var1, var2 and the merged set
at THE SAME positions. External VCFs are read-only.
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
import numpy as np, pathlib, sys
sys.stdout.reconfigure(encoding="utf-8")
EXT = work()
D = DEPOSIT
def scan(path, label):
    n_call = n_het = n_hom = 0
    per_pos = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#"): continue
            p = line.rstrip("\n").split("\t")
            gts = [x.split(":")[0] for x in p[9:]]
            c = h = 0
            for g in gts:
                if g in ("./.", ".", "./"): continue
                a = g.replace("|", "/").split("/")
                if len(a) != 2: continue
                c += 1
                if a[0] != a[1]: h += 1
            n_call += c; n_het += h; n_hom += c - h
            per_pos[(p[0], int(p[1]))] = (c, h)
    print("%s: called cells %d, heterozygous %d (%.4f %%)" % (label, n_call, n_het, 100 * n_het / max(n_call, 1)))
    return per_pos

v1 = scan(EXT / "6_новые данные_ответы + исходники +vcf" / "sunflower_var1_bam_merged_15_49.vcf", "var1")
v2 = scan(EXT / "6_новые данные_ответы + исходники +vcf" / "sunflower_var2_vcf_merged.vcf", "var2")

z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
contig = np.array([str(c) for c in z["chrom"]]); pos = np.asarray(z["pos"]).astype(np.int64)
call = np.sum(~np.isnan(G), 1); het = np.sum(G == 1, 1)
print("merged: called cells %d, heterozygous %d (%.4f %%)"
      % (int(call.sum()), int(het.sum()), 100 * het.sum() / call.sum()))

# at shared positions
keys = [(contig[i], int(pos[i])) for i in range(len(pos))]
both = [i for i, k in enumerate(keys) if k in v1 and k in v2]
print("\nshared positions var1 & var2 & merged: %d" % len(both))
c1 = sum(v1[keys[i]][0] for i in both); h1 = sum(v1[keys[i]][1] for i in both)
c2 = sum(v2[keys[i]][0] for i in both); h2 = sum(v2[keys[i]][1] for i in both)
cm = int(call[both].sum()); hm = int(het[both].sum())
print("  var1   : called %9d, het %7d (%.4f %%)" % (c1, h1, 100 * h1 / c1))
print("  var2   : called %9d, het %7d (%.4f %%)" % (c2, h2, 100 * h2 / c2))
print("  merged : called %9d, het %7d (%.4f %%)" % (cm, hm, 100 * hm / cm))
