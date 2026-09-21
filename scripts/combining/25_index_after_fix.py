# -*- coding: utf-8 -*-
"""
Paternal composite index, its bootstrap, and maternal GCA — before and after
correcting the shift in the year-means summary table (`scripts\\221_year_means_shift.py`).

The index definition is canonical, as in `25_index_canonical.py`:
z-standardize GCA over all 54 fathers (ddof = 1), reverse the emergence-to-flowering
direction, weights 0.35 / 0.30 / 0.10 / 0.00 / 0.10 / 0.15.

Bootstrap — over the 159 combinations of the summary table, 2,000 replicates, seed 20260817.
The same implementation is applied to both tables, so the difference between
“before” and “after” belongs to the data, not the implementation.
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
import os
import pandas as pd
import pathlib
import warnings

warnings.filterwarnings("ignore")

D = DEPOSIT
OUTD = pathlib.Path(os.environ.get("INDEX_OUT_DIR", str(OUT)))
OUTD.mkdir(parents=True, exist_ok=True)

TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter", "days_emergence_flowering"]
DIR = {"seed_yield": +1, "oil_content": +1, "seed_weight_1000": +1,
       "plant_height": 0, "head_diameter": +1, "days_emergence_flowering": -1}
W = {"seed_yield": 0.35, "oil_content": 0.30, "seed_weight_1000": 0.10,
     "plant_height": 0.00, "head_diameter": 0.10, "days_emergence_flowering": 0.15}
CONF = ["LI29", "LI30"]
NB, SEED = 2000, 20260817


def canon_index(g):
    z = g[TRAITS].copy()
    for t in TRAITS:
        z[t] = (z[t] - z[t].mean()) / z[t].std(ddof=1) * DIR[t]
    return sum(z[t] * w for t, w in W.items())


def gca_fathers(table):
    return pd.DataFrame({t: (lambda s: s - s.mean())(
        table.pivot_table(index="father", columns="mother", values=t).mean(1)) for t in TRAITS})


def gca_mothers(table, trait):
    s = table.pivot_table(index="father", columns="mother", values=trait).mean(0)
    return s - s.mean()


def bootstrap(table):
    pairs = table[["mother", "father"]].drop_duplicates().reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    c5, c1, done = {}, {}, 0
    for _ in range(NB):
        samp = pairs.iloc[rng.integers(0, len(pairs), len(pairs))]
        sub = samp.merge(table, on=["mother", "father"], how="left")
        g = gca_fathers(sub)
        if g.isna().any().any():
            continue
        s = canon_index(g).drop(index=[c for c in CONF if c in g.index], errors="ignore")
        s = s.sort_values(ascending=False)
        for f in s.head(5).index:
            c5[f] = c5.get(f, 0) + 1
        c1[s.index[0]] = c1.get(s.index[0], 0) + 1
        done += 1
    return (pd.DataFrame({"top5_pct": pd.Series({k: v / done * 100 for k, v in c5.items()}),
                          "first_pct": pd.Series({k: v / done * 100 for k, v in c1.items()})})
            .fillna(0).sort_values("top5_pct", ascending=False)), done


def baker_year_means(table, trait):
    """Baker ratio on the year-means table — a pooled estimate, as in `28_repro_162.py`.
    Reported in Results 3.1 as a comparison of two models (N-14)."""
    tab = table.pivot_table(index="father", columns="mother", values=trait)
    g = np.nanmean(tab.values)
    gf = np.nanmean(tab.values, 1) - g
    gm = np.nanmean(tab.values, 0) - g
    sca = tab.values - g - gf[:, None] - gm[None, :]
    vg = np.nanvar(np.concatenate([gf, gm]))
    vs = np.nanvar(sca.ravel())
    return 2 * vg / (2 * vg + vs)


def leave_one_trait_out(g):
    base = [k for k in canon_index(g).sort_values(ascending=False).index if k not in CONF][:5]
    out = {}
    for drop in [t for t in TRAITS if W[t] > 0]:
        z = g[TRAITS].copy()
        for t in TRAITS:
            z[t] = (z[t] - z[t].mean()) / z[t].std(ddof=1) * DIR[t]
        s = sum(z[t] * (0.0 if t == drop else W[t]) for t in TRAITS).sort_values(ascending=False)
        out[drop] = len(set([k for k in s.index if k not in CONF][:5]) & set(base))
    return base, out


TABLES = [("BEFORE correction", "delivered", en_table(pd.read_csv(INTERMEDIATE / "221_hybrid_means_delivered.csv"))),
          ("AFTER correction", "rebuilt", en_table(pd.read_csv(OUT / "221_hybrid_means_corrected.csv")))]

STAB = []
# Wave 16 (Sh8): leave-one-trait-out was printed only to the console, and the figure
# “2 of 5 without seed yield; 4–5 of 5 without any other trait” (registry row N-21)
# could not be reproduced from any deposited file. The calculation logic is unchanged —
# only a CSV write of the result was added.
LOTO = []
for label, tag, tab in TABLES:
    print("=" * 72)
    print(label)
    print("=" * 72)
    g = gca_fathers(tab)
    idx = canon_index(g).sort_values(ascending=False)
    top5 = [k for k in idx.index if k not in CONF][:5]
    print("  index, top-5: " + ", ".join(f"{k} {idx[k]:.4f} ({idx[k]:.3f})" for k in top5))
    print(f"  gap between the two leaders: {idx[top5[0]] - idx[top5[1]]:.4f}")
    b, done = bootstrap(tab)
    print(f"  bootstrap, {done} valid replicates of {NB}:")
    for k in top5:
        print(f"    {k:5s} top-5 {b.loc[k, 'top5_pct']:5.1f} %   first place {b.loc[k, 'first_pct']:5.1f} %")
    bb = b.copy()
    bb.index.name = "line"
    bb = bb.reset_index()
    bb.insert(0, "table", tag)
    bb.insert(2, "index_value", bb["line"].map(canon_index(g)))
    STAB.append(bb)
    nxt = [k for k in b.index if k not in top5][:2]
    print("    next after the top five: " +
          ", ".join(f"{k} {b.loc[k, 'top5_pct']:.1f} %" for k in nxt))
    base, loto = leave_one_trait_out(g)
    print("  leave-one-trait-out (how many of 5 positions were retained): " +
          ", ".join(f"{k} {v}" for k, v in loto.items()))
    for _drop, _kept in loto.items():
        LOTO.append({"table": tag, "trait_excluded": _drop, "trait_weight": W[_drop],
                     "retained_of_5": _kept, "base_five": " ".join(base)})
    print("  Baker on the year-means table (N-14): " + ", ".join(
        f"{t} {baker_year_means(tab, t):.4f}"
        for t in ("seed_yield", "oil_content", "seed_weight_1000")))
    print("  maternal GCA: " + " | ".join(
        f"{t}: " + ", ".join(f"{m}={v:+.2f}" for m, v in gca_mothers(tab, t).items())
        for t in ("oil_content", "seed_weight_1000", "plant_height")))
    print()

out = pd.concat(STAB, ignore_index=True)
out.to_csv(OUTD / "25_index_after_fix_stability.csv", index=False, encoding="utf-8-sig")
print("written:", OUTD / "25_index_after_fix_stability.csv")

loto_out = pd.DataFrame(LOTO)
loto_out.to_csv(OUTD / "25_index_leave_one_trait_out.csv", index=False, encoding="utf-8-sig")
print("written:", OUTD / "25_index_leave_one_trait_out.csv")
