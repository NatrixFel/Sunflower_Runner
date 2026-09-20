# -*- coding: utf-8 -*-
"""Can anything be estimated from a trait that has only plot means?

Days from emergence to flowering arrived in the project as a single value per
combination-year (318 rows = 159 combinations × 2 years); the plots themselves
were not retained. The question is what is lost.

Calibration. Take three traits that have BOTH plots AND means, collapse their
plots into combination-year means, and feed them into the same REML engine as
“one plot”. The difference between the resulting Baker ratio and the true
plot-level one shows the error incurred by working from means. The same
procedure is then applied to phenology.

Input:  deposit 22_hybrid_plot_level.parquet
       FINAL_rebutS/02_ДАННЫЕ/Фенотипы.xlsx  (sheet «Гибриды_средние»)
Output: 22_pheno_means_calibration.csv / .txt
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

HERE = Path(__file__).resolve().parent
REPO = work()
sys.path.insert(0, str(MODELS)); sys.dont_write_bytecode = True
from baker_v4 import reml_fit                                   # noqa: E402

OUT_CSV = OUT / "22_pheno_means_calibration.csv"
OUT_TXT = OUT / "fix06_pheno_means_calibration.txt"
PLOT = en_table(pd.read_parquet(OUT / "22_hybrid_plot_level.parquet"))
MO = {"M1": "VK101A", "M2": "VA761A", "M3": "VK934A"}

def fit(df, label, trait):
    d = (df.rename(columns={"mother": "mother", "father": "line", "year": "year",
                            "rep": "replicate", "trait": "trait",
                            "value": "value"})
           .dropna(subset=["value"]).reset_index(drop=True))
    f = reml_fit(d, with_yl=True, hess=False)
    s = f["s2"]
    return dict(trait=trait, input=label, n=len(d),
                s2_line=s["line"], s2_tester_x_line=s["tester_x_line"],
                s2_year_x_line=s["year_x_line"], s2_year_x_tester_x_line=s["year_x_tester_x_line"],
                s2_error=f["s2e"], baker=f["baker"])

rows = []
for tr in ["seed_yield", "oil_content", "seed_weight_1000", "plant_height", "head_diameter"]:
    d = PLOT[PLOT["trait"] == tr]
    rows.append(fit(d, "plots", tr))
    m = (d.groupby(["mother", "father", "year", "trait"], as_index=False)["value"].mean())
    m["rep"] = 1
    rows.append(fit(m, "combination_year_means", tr))

gm = en_table(pd.read_excel(REPO / "FINAL_rebutS" / "02_ДАННЫЕ" / "Фенотипы.xlsx", sheet_name="Гибриды_средние"))
ph = gm[gm["trait"] == "days_emergence_flowering"].copy()
ph = pd.DataFrame(dict(mother=ph["mother"].map(MO), father=ph["line"], year=ph["year"],
                       rep=1, trait="days_emergence_flowering", value=ph["value"]))
rows.append(fit(ph, "combination_year_means", "days_emergence_flowering"))

T = pd.DataFrame(rows)
T.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

txt = ["What working from means instead of plots gives", "",
       f"{'trait':16s}{'input':26s}{'n':>6s}{'baker':>10s}{'sigma2_line':>11s}"
       f"{'sigma2_tester_x_line':>11s}{'s2_error':>12s}"]
for _, r in T.iterrows():
    txt.append(f"{r['trait']:16s}{r['input']:26s}{r['n']:6d}{r['baker']:10.4f}"
               f"{r['s2_line']:11.4f}{r['s2_tester_x_line']:11.4f}{r['s2_error']:12.4f}")
txt.append("")
w = T.pivot_table(index="trait", columns="input", values="baker")
w = w.dropna()
w["shift"] = w["combination_year_means"] - w["plots"]
txt.append("Baker ratio shift when moving from plots to means:")
for i, r in w.iterrows():
    txt.append(f"  {i:16s} {r['plots']:.4f} -> {r['combination_year_means']:.4f}"
               f"   ({r['shift']:+.4f})")
txt += ["",
        f"Phenology (means only): Baker = "
        f"{float(T[T['trait']=='days_emergence_flowering']['baker'].iloc[0]):.4f}",
        "The means-based estimate carries a shift of the same order as shown above",
        "on the five traits with a known answer; plot-level repeatability for",
        "phenology is not estimated at all — residual mixes Year×Tester×Line and trial error."]
OUT_TXT.write_text("\n".join(txt), encoding="utf-8")
print("\n".join(txt))
