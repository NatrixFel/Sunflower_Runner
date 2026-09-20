# -*- coding: utf-8 -*-
"""Does the undocumented substitution “451 seeds from 3 plants” affect line H².

In the VNIIMK workbook (2022 sheet, self-pollination block, LI27) there is a text cell
“451 (from 3 plants)”. In 22_lines_tidy.parquet it is replaced by three IDENTICAL values
150.333 = 451/3, i.e. one aggregated record is expanded into three observations.
H² is compared on the published table and on the table where those three copies are
collapsed to a single record.
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
import sys, pathlib, numpy as np, pandas as pd
sys.dont_write_bytecode = True
R = DEPOSIT
sys.path.insert(0, str(R / "scripts"))
from baker_v4 import REML, dummies

L = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
mask = ((L["trait"] == "autofertility_self") & (L["year"] == 2022) & (L["genotype"] == "LI27")
        & (np.isclose(L["value"], 451 / 3)))
L2 = pd.concat([L[~mask], L[mask].iloc[:1]], ignore_index=True)

def h2(df, trait):
    d = df[(df["trait"] == trait) & (~df["genotype"].isin(["LI29", "LI30"]))].dropna(subset=["value"])
    y = d["value"].to_numpy(float)
    yr = d["year"].astype(str).tolist(); gen = d["genotype"].astype(str).tolist()
    gy = [a + "|" + b for a, b in zip(gen, yr)]
    X = np.hstack([np.ones((len(y), 1)), dummies(yr)])
    ny, ng = d["year"].nunique(), d["genotype"].nunique()
    npl = len(y) / (ng * ny)
    f = REML(y, X, [dummies(gen), dummies(gy)], ["G", "GY"]).fit(
        [[0.3, 0.3], [0.05, 0.05], [1.0, 0.3], [3.0, 1.0]])
    g, gyv, e = f["s2"]["G"], f["s2"]["GY"], f["s2e"]
    return len(y), g / (g + gyv / ny + e / (ny * npl)), g, gyv, e

print(f"copies of 150.333 in the table: {int(mask.sum())}")
for lab, df in [("as in the paper", L), ("one record instead of three", L2)]:
    for t in ["autofertility_self", "autofertility_open"]:
        n, H, g, gy, e = h2(df, t)
        print(f"  {lab:24s} {t}: n={n:4d}  H²={H:.4f}  s2g={g:.0f}  s2gy={gy:.0f}  s2e={e:.0f}")
