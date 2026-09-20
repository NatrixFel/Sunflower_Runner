# -*- coding: utf-8 -*-
"""Cell counts for the var1+var2 merge rule (Methods 2.3).

Two position-index conventions, selected with --convention:

  last   production: last occurrence of a duplicated key, as in 23_merge_vcfs.py
         (2,250 allele-mismatch sites). Default.
  first  np.intersect1d first occurrence (1,772 allele-mismatch sites).

Reads the undeposited raw call sets under SUNFLOWER_WORK.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import numpy as np, allel

parser = argparse.ArgumentParser()
parser.add_argument("--convention", choices=("first", "last"), default="last")
args = parser.parse_args()

sys.stdout.reconfigure(encoding="utf-8")
ROOT = work()
NEW = ROOT / "6_новые данные_ответы + исходники +vcf"
V1 = NEW / "sunflower_var1_bam_merged_15_49.vcf"
V2 = NEW / "sunflower_var2_vcf_merged.vcf"

SEQ = {}
for n in range(1, 33): SEQ[n] = "LI%d" % n
SEQ[33] = "LI34"; SEQ[34] = "LI35"
for n in range(35, 53): SEQ[n] = "LI%d" % (n + 2)
SEQ[53] = "LI33"; SEQ[54] = "LI36"; SEQ[55] = "VA761"; SEQ[56] = "VK101"; SEQ[57] = "VK934"
canon = [SEQ[s] for s in range(1, 58)]


def disc_consensus(reps):
    v0 = (reps == 0).sum(1); v1 = (reps == 1).sum(1); v2 = (reps == 2).sum(1)
    mx = np.stack([v0, v1, v2], 1).max(1); has = mx > 0
    out = np.full(reps.shape[0], -1, dtype=np.int8)
    out = np.where(has & (v0 == mx), 0, out)
    out = np.where(has & (out == -1) & (v2 == mx), 2, out)
    out = np.where(has & (out == -1) & (v1 == mx), 1, out)
    return out.astype(np.int8)


def load_disc(path, is_var1):
    print("  reading %s…" % path.name, flush=True)
    cb = allel.read_vcf(str(path), fields=["samples", "variants/CHROM", "variants/POS",
                                           "variants/REF", "variants/ALT", "calldata/GT"])
    ga = allel.GenotypeArray(cb["calldata/GT"]); nalt = ga.to_n_alt(fill=-1).astype(np.int8)
    sr = en_ids(cb["samples"].astype(str)); ci = {n: i for i, n in enumerate(sr)}
    disc = np.full((nalt.shape[0], 57), -1, dtype=np.int8)
    stats = {}
    if is_var1:
        r15 = [i for i, s in enumerate(sr) if s.startswith("15-")]
        r49 = [i for i, s in enumerate(sr) if s.startswith("49-")]
        print("    replicates: LI15 %d columns, LI51 %d columns" % (len(r15), len(r49)))
        c15 = disc_consensus(nalt[:, r15]); c49 = disc_consensus(nalt[:, r49])
        for nm, reps in (("LI15", nalt[:, r15]), ("LI51", nalt[:, r49])):
            called = reps >= 0
            n_called = called.sum(1)
            disagree = np.zeros(reps.shape[0], dtype=bool)
            for a in range(reps.shape[1]):
                for b in range(a + 1, reps.shape[1]):
                    both = (reps[:, a] >= 0) & (reps[:, b] >= 0)
                    disagree |= both & (reps[:, a] != reps[:, b])
            stats[nm] = (int((n_called > 0).sum()), int(disagree.sum()), reps.shape[1])
        for j, nm in enumerate(canon):
            if nm == "LI15": disc[:, j] = c15
            elif nm == "LI51": disc[:, j] = c49
            else:
                seq = [k for k, v in SEQ.items() if v == nm][0]
                disc[:, j] = nalt[:, ci["%d.0" % seq]]
    else:
        for j, nm in enumerate(canon):
            seq = [k for k, v in SEQ.items() if v == nm][0]
            key = str(seq)
            disc[:, j] = nalt[:, ci[key]] if key in ci else -1
    return (cb["variants/CHROM"], cb["variants/POS"].astype(np.int64),
            cb["variants/REF"], cb["variants/ALT"][:, 0], disc, stats)


c1, p1, ref1, alt1, d1, st1 = load_disc(V1, True)
c2, p2, ref2, alt2, d2, st2 = load_disc(V2, False)
chroms = sorted(set(c1) | set(c2)); cidx = {c: i for i, c in enumerate(chroms)}
K1 = np.array([cidx[c] for c in c1], dtype=np.int64) * 10**9 + p1
K2 = np.array([cidx[c] for c in c2], dtype=np.int64) * 10**9 + p2

if args.convention == "first":
    common, i1, i2 = np.intersect1d(K1, K2, return_indices=True)
    print("\nshared positions (first occurrence): %d" % len(common))
else:
    pos1 = {int(k): i for i, k in enumerate(K1)}
    pos2 = {int(k): i for i, k in enumerate(K2)}
    print("\nduplicate positions: var1 %d, var2 %d"
          % (len(K1) - len(pos1), len(K2) - len(pos2)))
    shared = [int(k) for k in np.union1d(K1, K2) if int(k) in pos1 and int(k) in pos2]
    i1 = np.array([pos1[k] for k in shared]); i2 = np.array([pos2[k] for k in shared])
    print("union: %d sites" % len(np.union1d(K1, K2)))
    print("shared positions (last occurrence): %d" % len(shared))

same = (ref1[i1] == ref2[i2]) & (alt1[i1] == alt2[i2])
print("of which different REF/ALT (var1 kept): %d" % int((~same).sum()))
print("of which matching alleles: %d" % int(same.sum()))

g1, g2 = d1[i1[same]], d2[i2[same]]
m1, m2 = g1 < 0, g2 < 0
both = ~m1 & ~m2
eq = both & (g1 == g2)
hom1 = (g1 == 0) | (g1 == 2); hom2 = (g2 == 0) | (g2 == 2)
hom_over_het = both & ~eq & ((hom1 & ~hom2) | (hom2 & ~hom1))
conflict = both & ~eq & hom1 & hom2
tot = g1.size
print("\nCELLS (%d positions × 57 = %d):" % (g1.shape[0], tot))
for nm, mask in (("both called and equal", eq),
                 ("hom vs het → hom", hom_over_het),
                 ("two different homs → skip", conflict),
                 ("called only in var1", ~m1 & m2),
                 ("called only in var2", m1 & ~m2),
                 ("missing in both", m1 & m2)):
    print("  %-40s %12d  %6.3f %%" % (nm, int(mask.sum()), 100 * mask.sum() / tot))

print("\nvar1 replicate consolidation:")
for nm, (sites, dis, ncol) in st1.items():
    print("  %-6s columns %d, sites called %8d, disagreed %8d (%.2f %%)"
          % (nm, ncol, sites, dis, 100 * dis / max(1, sites)))
