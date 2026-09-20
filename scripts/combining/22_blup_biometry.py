# -*- coding: utf-8 -*-
"""EXTENSION. Hybrid BLUP phenotypes for height and diameter on the corrected
plot-level table (162 combinations) — previously impossible.

The model is the same as for the three productivity traits in 22_parse_hybrid_reps.py:
    value ~ C(year) + C(year):C(rep),  combination is a random effect, REML;
    BLUP phenotype = grand mean + combination random effect.

The published four columns (seed yield, oil content, 1000-seed weight, oil yield)
are copied from 22_hybrid_blup_phenotypes.csv WITHOUT changes; two new ones are added.

Input:  deposit 22_hybrid_plot_level.parquet
        deposit 22_hybrid_blup_phenotypes.csv
Output: 22_hybrid_blup_phenotypes_ext.csv
        22_blup_biometry.txt
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
from pathlib import Path
import warnings, numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")

PARQ = OUT / "22_hybrid_plot_level.parquet"
OLD  = OUT / "22_hybrid_blup_phenotypes.csv"
OUT_CSV = OUT / "22_hybrid_blup_phenotypes_ext.csv"
TXT  = OUT / "22_blup_biometry.txt"

P = en_table(pd.read_parquet(PARQ))
bl = en_table(pd.read_csv(OLD, encoding="utf-8-sig"))
txt = ["Hybrid BLUP phenotypes: extension to height and diameter", ""]
new = {}
for trait in ["plant_height", "head_diameter", "seed_yield", "oil_content", "seed_weight_1000"]:
    d = P[P["trait"] == trait].copy()
    d["g"] = d["mother"] + "_" + d["father"]
    # Year + Rep(Year) is encoded as ONE “year-plot” factor: the same linear
    # span, but full rank. The literal C(year)+C(year):C(rep) is singular
    # when the number of plots differs by year (2 in 2022, 3 in 2023) — the
    # same singularity as defect B1 in baker_v3.
    d["yr_rep"] = d["year"].astype(str) + "_" + d["rep"].astype(str)
    mf = smf.mixedlm("value ~ C(yr_rep)", d, groups=d["g"]).fit(reml=True)
    s2g = float(mf.cov_re.iloc[0, 0]); s2e = float(mf.scale)
    gm = d["value"].mean()
    vals = {g: gm + float(v.iloc[0]) for g, v in mf.random_effects.items()}
    r = len(d) / d["g"].nunique()
    txt.append(f"{trait:14s} n={len(d):4d} combinations={d['g'].nunique():3d}  "
               f"s2g={s2g:8.3f} s2e={s2e:8.3f}  repeatability={s2g/(s2g+s2e):.3f}  "
               f"H2(mean)={s2g/(s2g+s2e/r):.3f}")
    if trait in ("plant_height", "head_diameter"):
        new[trait] = pd.Series(vals)
    else:                                   # check: do they match the published columns
        chk = bl.set_index("hybrid")[trait].reindex(pd.Series(vals).index)
        dd = (chk - pd.Series(vals)).abs()
        txt.append(f"{'':14s} check vs published column: max |Δ| = {dd.max():.4f} "
                   f"(correlation {np.corrcoef(chk.dropna(), pd.Series(vals)[chk.dropna().index])[0,1]:.6f})")

EXT = bl.set_index("hybrid").copy()
for t, s in new.items():
    EXT[t] = s
EXT = EXT.reset_index()
EXT.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
txt += ["", f"written: {OUT.name}, {len(EXT)} hybrids × {EXT.shape[1]-1} traits",
        f"columns: {', '.join(EXT.columns[1:])}",
        f"non-missing in new columns: plant_height {EXT['plant_height'].notna().sum()}, "
        f"head_diameter {EXT['head_diameter'].notna().sum()}"]
TXT.write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
