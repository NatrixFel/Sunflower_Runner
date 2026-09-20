"""
CANONICAL BLINK on paternal lines — replacement for the retracted counters
BLINK 266/791/256, computed on var2 with reference substitution.

Panel — merged (cr>=0.9, MAF>=0.05), covariates — three principal components on
centered and standardized genotypes (canonical D-3-ter). A centering-only
variant is also computed to assess sensitivity to preprocessing.

Output: 26_canonical_blink.csv
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
import sys
import numpy as np, pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
sys.path.insert(0, str(ROOT / "phase8_blink_python"))
from blink import blink

HERE = OUT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TRAITS = ["oil_content", "seed_weight_1000", "hull_content", "head_diameter", "plant_height",
          "days_emergence_flowering", "days_central_to_side", "autofertility_self", "autofertility_open"]

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
pos = z["pos"]; samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
ALLF = [s for s in samples if is_line_id(s)]

ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

rows = []
for tag, lines in (("54_with_confectionery", ALLF),
                   ("52_oilseed_only", [f for f in ALLF if f not in CONF])):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2
    maf = np.minimum(af, 1 - af)
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
    Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
    chf = chrn[keep].astype(int); pf = pos[keep]
    X0 = Gi.T.astype(float)
    def make_pc(standardize):
        Xc = X0 - X0.mean(0)
        if standardize: Xc = Xc / (X0.std(0) + 1e-9)
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        return U[:, :3] * S[:3]
    print(f"\n### {tag}: {len(lines)} lines, {Gi.shape[0]:,} SNP", flush=True)
    for model, PC in (("3_PC_standardised_canon", make_pc(True)),
                      ("3_PC_centred", make_pc(False))):
        tot = 0; detail = []
        for tr in TRAITS:
            if tr not in mean_ph.columns: continue
            y = mean_ph[tr].reindex(lines).values.astype(float)
            res = blink(y, Gi, chf, pf, PCs=PC)
            p = res["pvals"]
            k = int(np.nansum(p < 5e-8))
            tot += k
            if k: detail.append(f"{tr} {k}")
            rows.append({"panel": tag, "model": model, "trait": tr,
                         "n_SNP": Gi.shape[0], "BLINK_n_p<5e-8": k,
                         "n_pseudo_QTN": len(res.get("pseudo_qtn", []))})
        print(f"  {model:30s} total at 5e-8: {tot:4d}"
              + (f"   ({', '.join(detail)})" if detail else ""), flush=True)

df = pd.DataFrame(rows)
df.to_csv(HERE / "26_canonical_blink.csv", index=False, encoding="utf-8-sig")
print("\n=== BLINK SUMMARY ===")
print(df.groupby(["model", "panel"])["BLINK_n_p<5e-8"].sum().to_string())
print("\nWritten: 26_canonical_blink.csv")
