# -*- coding: utf-8 -*-
"""F3 / S2. Re-run of the BLINK self-test on the ANALYSIS PANEL.

The previous “10 of 10” result was obtained on `phase8_blink_python\\50_matrix_54.npz`
(an early genotype matrix), not on the analysis panel. Here the same self-test —
same simulation parameters, same seed — is run on the analysis panel: merged set,
chr 1–17, call rate ≥ 0.9, MAF ≥ 0.05, 54 paternal lines, mean imputation.

The criterion of the previous self-test in `blink.py` is that the causal SNP or a
marker in the ±1 Mb LD window falls in the **top-5**. The Supplementary text writes
“recovered at rank 1”, so both criteria are counted separately here.

Edits nothing; only prints and writes CSV.
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
import sys, pathlib
import numpy as np, pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = DEPOSIT
D = DEPOSIT
SCR = OUT
EXT = DEPOSIT
from blink import blink  # noqa: E402

CONF = ("LI29", "LI30")

def cn(c):
    try:
        return int(str(c).replace("CM00", "").replace(".2", "")) - 7889
    except Exception:
        return -1

def analysis_panel():
    z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
    G = z["genotypes"].astype(float); G[G < 0] = np.nan
    samp = en_ids(z["samples"].astype(str))
    chrom = np.array([cn(c) for c in z["chrom"]])
    pos = np.asarray(z["pos"]).astype(np.int64)
    lines = [s for s in samp if is_line_id(s)]
    idx = [samp.index(l) for l in lines]
    Gl = G[:, idx]
    cr = np.mean(~np.isnan(Gl), 1)
    keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9)
    af = np.nanmean(Gl, 1) / 2
    keep &= np.minimum(af, 1 - af) >= 0.05
    Gp = Gl[keep]
    Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
    return Gi, chrom[keep], pos[keep], lines

def selftest(Gi, chrom, pos, label, trials=10, seed=42):
    m, n = Gi.shape
    rng = np.random.RandomState(seed)
    rows = []
    print("=" * 78)
    print("%s: %d SNP × %d lines" % (label, m, n))
    for t in range(trials):
        causal = rng.randint(0, m)
        g = Gi[causal]
        if np.std(g) < 1e-6:
            continue
        gz = (g - g.mean()) / g.std()
        poly = Gi[rng.choice(m, 30, replace=False)].mean(axis=0)
        poly = (poly - poly.mean()) / (poly.std() + 1e-9)
        y = 2.0 * gz + 0.5 * poly + rng.normal(0, 1.0, n)
        res = blink(y, Gi, chrom, pos, PCs=None, verbose=False)
        p = np.where(np.isnan(res["pvals"]), np.inf, res["pvals"])
        order = np.argsort(p)
        rank = int(np.where(order == causal)[0][0]) + 1
        def in_window(i):
            return (chrom[i] == chrom[causal]) and (abs(pos[i] - pos[causal]) < 1_000_000)
        top1, top5 = order[:1], order[:5]
        hit1 = (causal in top1) or any(in_window(i) for i in top1)
        hit5 = (causal in top5) or any(in_window(i) for i in top5)
        # rank of the best marker in the LD window
        win = [i for i in order if in_window(i)]
        rank_win = int(np.where(order == win[0])[0][0]) + 1 if win else -1
        rows.append(dict(trial=t + 1, chr=int(chrom[causal]), pos=int(pos[causal]),
                         rank_causal=rank, rank_window=rank_win,
                         p_causal=float(res["pvals"][causal]),
                         n_pseudo=len(res["pseudo_qtn"]),
                         hit_rank1=bool(hit1), hit_top5=bool(hit5)))
        print("  trial %2d: chr%d:%d  causal rank=%-6d window rank=%-6d p=%.2e "
              "|pseudo|=%d  rank1=%s top5=%s"
              % (t + 1, chrom[causal], pos[causal], rank, rank_win,
                 res["pvals"][causal], len(res["pseudo_qtn"]),
                 "✓" if hit1 else "✗", "✓" if hit5 else "✗"))
    df = pd.DataFrame(rows)
    print("  TOTAL %s: rank 1 — %d of %d; top-5 — %d of %d"
          % (label, df["hit_rank1"].sum(), len(df), df["hit_top5"].sum(), len(df)))
    return df

if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "brain" / "26_supp_s2_selftest.csv"

    Gi, chrom, pos, lines = analysis_panel()
    df_new = selftest(Gi, chrom, pos, "ANALYSIS PANEL (call rate ≥ 0.9, MAF ≥ 0.05)")
    df_new["panel"] = "analysis_panel"

    # for comparison — the previous matrix on which the published 10/10 was obtained
    old = EXT / "phase8_blink_python" / "50_matrix_54.npz"
    if old.exists():
        d = np.load(old, allow_pickle=True)
        G = d["G"].astype(float)
        Go = np.where(np.isnan(G), np.nanmean(G, axis=1, keepdims=True), G)
        df_old = selftest(Go, np.asarray(d["chr"]), np.asarray(d["pos"]),
                          "PREVIOUS MATRIX 50_matrix_54.npz")
        df_old["panel"] = "matrix_54"
        df = pd.concat([df_new, df_old], ignore_index=True)
    else:
        print("previous matrix not found — comparison is not possible")
        df = df_new
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print("\nwritten:", out)
