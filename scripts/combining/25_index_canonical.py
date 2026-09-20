"""
Paternal index STRICTLY by the canonical definition in 25_gca_sca.py:
  - z-standardize GCA over ALL 54 fathers (confectionery lines are NOT dropped before z);
  - DIRECTION: days_emergence_flowering = -1, plant_height = 0, the rest +1;
  - WEIGHTS: yield 0.35, oil 0.30, seed weight 0.10, height 0.00, diameter 0.10, flowering 0.15.
Step 1 — reconcile with the saved 25_paternal_selection_index.csv.
Step 2 — bootstrap over hybrids with THIS definition.
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
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
HERE = OUT
CONF = ["LI29", "LI30"]
TRAITS = ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter", "days_emergence_flowering"]
DIRECTION = {"seed_yield": +1, "oil_content": +1, "seed_weight_1000": +1,
             "plant_height": 0, "head_diameter": +1, "days_emergence_flowering": -1}
WEIGHTS = {"seed_yield": 0.35, "oil_content": 0.30, "seed_weight_1000": 0.10,
           "plant_height": 0.00, "head_diameter": 0.10, "days_emergence_flowering": 0.15}

def canon_index(gca_df):
    """gca_df: index=father, columns=TRAITS. Exact copy of the phase3c logic."""
    z = gca_df[TRAITS].copy()
    for t in TRAITS:
        z[t] = (z[t] - z[t].mean()) / z[t].std(ddof=1) * DIRECTION[t]
    return sum(z[t] * w for t, w in WEIGHTS.items())

# ---------- step 1: reconcile ----------
gca = en_table(pd.read_csv(OUT / "25_gca_fathers.csv")).set_index("father")
idx = canon_index(gca).sort_values(ascending=False)
saved = en_table(pd.read_csv(INTERMEDIATE / "25_paternal_selection_index.csv")).set_index("father")["selection_index"]
cmp = pd.concat([idx.rename("ours"), saved.rename("saved")], axis=1).dropna()
cmp["difference"] = (cmp["ours"] - cmp["saved"]).abs()
print("RECONCILIATION WITH THE CANONICAL phase3c OUTPUT:")
print(f"  maximum discrepancy: {cmp['difference'].max():.2e}")
print(f"  matches to 10 digits: {bool(cmp['difference'].max() < 1e-10)}")
print("\n  top-8 (canonical):")
for k, v in idx.head(8).items():
    print(f"    {k:6s} {v:+.4f}")

top5_canon = [k for k in idx.index if k not in CONF][:5]
print(f"\n  canonical top-5 (excluding confectionery): {top5_canon}")

# ---------- step 2: bootstrap over hybrids with the canonical definition ----------
h = en_table(pd.read_parquet(INTERMEDIATE / "01_hybrids_tidy.parquet"))
KEY = "mother_long" if "mother_long" in h.columns else "mother"
h = h[h["trait"].isin(TRAITS)].copy()
print(f"\nbootstrap: combinations={h.groupby([KEY,'father']).ngroups}, fathers={h['father'].nunique()}")

def gca_from(df):
    rows = {}
    for t in TRAITS:
        sub = df[df["trait"] == t]
        if not len(sub): return None
        tab = sub.groupby(["father", KEY])["value"].mean().unstack()
        g = tab.mean(1)
        rows[t] = g - g.mean()
    return pd.DataFrame(rows)

base_g = gca_from(h)
base_i = canon_index(base_g).sort_values(ascending=False)
print("  index from raw phase1 (convergence check), top-6:",
      ", ".join(f"{k}={v:+.3f}" for k, v in base_i.head(6).items()))

pairs = h[[KEY, "father"]].drop_duplicates().reset_index(drop=True)
rng = np.random.default_rng(20260817)
NB = 2000
c5, c3, c1 = {}, {}, {}
for _ in range(NB):
    samp = pairs.iloc[rng.integers(0, len(pairs), len(pairs))]
    sub = samp.merge(h, on=[KEY, "father"], how="left")
    g = gca_from(sub)
    if g is None or g.isna().any().any(): continue
    s = canon_index(g).drop(index=[c for c in CONF if c in g.index], errors="ignore")
    s = s.sort_values(ascending=False)
    for f in s.head(5).index: c5[f] = c5.get(f, 0) + 1
    for f in s.head(3).index: c3[f] = c3.get(f, 0) + 1
    if len(s): c1[s.index[0]] = c1.get(s.index[0], 0) + 1

res = pd.DataFrame({
    "index": idx,
    "top5_pct": pd.Series({k: v/NB*100 for k, v in c5.items()}),
    "top3_pct": pd.Series({k: v/NB*100 for k, v in c3.items()}),
    "first_pct": pd.Series({k: v/NB*100 for k, v in c1.items()}),
}).fillna(0)
res = res.drop(index=[c for c in CONF if c in res.index], errors="ignore")
res = res.sort_values("top5_pct", ascending=False)
print(f"\nBOOTSTRAP ({NB} replicates), canonical index:")
print(res.head(9).round(1).to_string())

# leave-one-trait-out on the canonical definition
print("\nLEAVE-ONE-TRAIT-OUT (canonical index):")
print(f"  {'full':26s} -> {top5_canon}")
for drop in [t for t in TRAITS if WEIGHTS[t] > 0]:
    W2 = {k: (0.0 if k == drop else v) for k, v in WEIGHTS.items()}
    z = gca[TRAITS].copy()
    for t in TRAITS:
        z[t] = (z[t] - z[t].mean()) / z[t].std(ddof=1) * DIRECTION[t]
    s = sum(z[t] * w for t, w in W2.items()).sort_values(ascending=False)
    s = [k for k in s.index if k not in CONF][:5]
    print(f"  without «{drop:16s}» -> {s}  (overlap {len(set(s)&set(top5_canon))}/5)")

res.to_csv(HERE / "index_canonical_stability.csv", encoding="utf-8-sig")
print("\nSaved: index_canonical_stability.csv")
