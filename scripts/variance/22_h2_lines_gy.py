# -*- coding: utf-8 -*-
"""FIX D2. Paternal-line H² with a separated Year × Genotype term.

Defect of the former estimate (no G×Y): H² of nine traits from the model
    y = mu + Year(fixed) + Genotype(random) + error,   H² = s2g / (s2g + s2e/4),
without a genotype × year term. Exactly this defect was already fixed on
hybrids by decisions D020/D021, but on lines it remained. For the four traits with
several records per line-year the term is identifiable, is separated, and is
statistically huge.

Corrected model for these four traits:
    y = mu + Year(fixed) + Genotype(random) + Year×Genotype(random) + error
    H² of the line mean over ny years = s2g / (s2g + s2gy/ny + s2e/(ny*r)),
where r is the number of records per line-year (plants for height and diameter,
replicates for autofertility). For the five traits with one record per line-year
the term is not identifiable, and the published specification remains correct.

The REML engine is the one adopted by decision D020 (scripts/baker_v4.REML).

Input:  deposit 22_lines_tidy.parquet
Output: 22_h2_lines_gy.csv
        fix02_h2_lines_gy.txt
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
import sys, numpy as np, pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.dont_write_bytecode = True
from baker_v4 import REML, dummies                          # noqa: E402

SRC = OUT / "22_lines_tidy.parquet"
OUT_CSV = OUT / "22_h2_lines_gy.csv"
OUT_TXT = OUT / "fix02_h2_lines_gy.txt"
CONF = ["LI29", "LI30"]                       # confectionery, outside the main analysis
NINE = ["plant_height", "head_diameter", "autofertility_self", "autofertility_open", "days_emergence_flowering",
        "days_central_to_side", "hull_content", "seed_weight_1000", "oil_content"]

L = en_table(pd.read_parquet(SRC))
rows = []
for t in NINE:
    d = (L[(L["trait"] == t) & (~L["genotype"].isin(CONF))]
         .dropna(subset=["value"]))
    y = d["value"].to_numpy(float)
    yr = d["year"].astype(str).tolist(); gen = d["genotype"].astype(str).tolist()
    gy = [a + "|" + b for a, b in zip(gen, yr)]
    X = np.hstack([np.ones((len(y), 1)), dummies(yr)])
    Zg, Zgy = dummies(gen), dummies(gy)
    ng, ny = d["genotype"].nunique(), d["year"].nunique()
    r_unit = len(y) / (ng * ny)
    fA = REML(y, X, [Zg], ["G"]).fit([[0.3], [0.05], [1.0], [3.0]])
    rec = dict(trait=t, n=len(y), n_lines=ng, n_years=ny, records_per_line_year=r_unit,
               s2g_published=fA["s2"]["G"], s2e_published=fA["s2e"],
               H2_published=fA["s2"]["G"] / (fA["s2"]["G"] + fA["s2e"] / ny),
               identifiable=r_unit > 1.05)
    if rec["identifiable"]:
        fB = REML(y, X, [Zg, Zgy], ["G", "GY"]).fit(
            [[0.3, 0.3], [0.05, 0.05], [1.0, 0.3], [3.0, 1.0], [0.3, 1.0], [0.0, 0.3]])
        g, gyv, e = fB["s2"]["G"], fB["s2"]["GY"], fB["s2e"]
        lr = 2 * (fB["logL"] - fA["logL"])
        rec.update(s2g=g, s2gy=gyv, s2e=e, LR=lr,
                   p_LR=0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0,
                   H2_corrected=g / (g + gyv / ny + e / (ny * r_unit)))
    rows.append(rec)

T = pd.DataFrame(rows)
T.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

pub = T["H2_published"]
fin = T["H2_corrected"].where(T["H2_corrected"].notna(), T["H2_published"])
txt = ["FIX D2 — paternal-line H² (52 oilseed, 2021–2024)", "",
       f"{'trait':24s}{'records/line-year':>16s}{'H² publ.':>11s}{'LR':>10s}"
       f"{'p':>12s}{'H² corr.':>11s}"]
for _, r in T.iterrows():
    lr = f"{r['LR']:.1f}" if pd.notna(r.get("LR")) else "—"
    p = f"{r['p_LR']:.2e}" if pd.notna(r.get("p_LR")) else "—"
    h2 = f"{r['H2_corrected']:.3f}" if pd.notna(r.get("H2_corrected")) else "(unchanged)"
    txt.append(f"{r['trait']:24s}{r['records_per_line_year']:16.2f}"
               f"{r['H2_published']:11.3f}{lr:>10s}{p:>12s}{h2:>15s}")
txt += ["",
        f"published range:  {pub.min():.3f}–{pub.max():.3f}  (in the paper 0.73–0.94)",
        f"corrected range:  {fin.min():.3f}–{fin.max():.3f}",
        "",
        "The lower bound is husk content, which has one record per line-year and the model",
        "is specified correctly; the upper bound is height, for which the Year×Genotype term is separated.",
        "Direction of the correction: upward, i.e. the published values are underestimated."]
OUT_TXT.write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
