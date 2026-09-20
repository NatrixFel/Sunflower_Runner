"""Use-type artefact vs PCA covariates (merged panel, EMMAX).

Two recorded runs, selected with --mode:

  freeze  NBOOT=2000, seed 20260817, covariates = none / 3 PC standardised;
          writes minp_*.npy for 26_threshold_uncertainty.py and pca_freeze_{panel}.csv
  full    NBOOT=600,  seed 4242,     covariates add the centred-only 3 PC level;
          writes pca_full_{panel}.csv

Run `python 26_pca_artifact.py [54|52] --mode freeze|full`.
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
import argparse
import numpy as np, pandas as pd
from scipy import stats, optimize
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("panel", nargs="?", default="54")
parser.add_argument("--mode", choices=("freeze", "full"), default="freeze")
args = parser.parse_args()
PANEL = args.panel
MODE = args.mode

ROOT = DEPOSIT
HERE = OUT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TRAITS = ["oil_content","hull_content","seed_weight_1000","head_diameter","plant_height","days_emergence_flowering","autofertility_self","autofertility_open"]
NBOOT = 2000 if MODE == "freeze" else 600
SEED = 20260817 if MODE == "freeze" else 4242

z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
ALLF = [s for s in samples if is_line_id(s)]
fathers = ALLF if PANEL == "54" else [f for f in ALLF if f not in CONF]
col = [samples.index(l) for l in fathers]
G = nalt[:, col]
cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
print(f"panel: {Gi.shape[0]:,} SNP x {len(fathers)} fathers  mode={MODE}")

p_al = Gi.mean(1) / 2; W = Gi - 2 * p_al[:, None]
K = (W.T @ W) / (2 * np.sum(p_al * (1 - p_al)))

X0 = Gi.T.astype(float)
isconf = np.array([f in CONF for f in fathers], float)

def make_pc(standardize):
    Xc = X0 - X0.mean(0)
    if standardize:
        Xc = Xc / (X0.std(0) + 1e-9)
    U, S, _ = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :3] * S[:3]

for lab, st in (("centred", False), ("standardised", True)):
    pc = make_pc(st)
    r = abs(np.corrcoef(pc[:, 0], isconf)[0, 1])
    print(f"  PC1 ({lab:14s}) |corr| with confectionery indicator = {r:.3f}")

ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

def emmax(y, Gm, Kk, PC=None):
    m = ~np.isnan(y); yv = y[m]; Ks = Kk[np.ix_(m, m)]; Gs = Gm[:, m]
    ev, U = np.linalg.eigh(Ks); ev = np.maximum(ev, 1e-9)
    X = np.ones((len(yv), 1))
    if PC is not None: X = np.column_stack([X, PC[m]])
    yr = U.T @ yv; Xr = U.T @ X
    def nll(t):
        sg, se = np.exp(t); D = sg * ev + se
        Wd = 1 / D; XtWX = Xr.T @ (Xr * Wd[:, None])
        try: inv = np.linalg.inv(XtWX)
        except Exception: return 1e10
        b = inv @ (Xr.T @ (yr * Wd)); rr = yr - Xr @ b
        _, ld = np.linalg.slogdet(XtWX)
        return 0.5 * (np.log(D).sum() + (rr ** 2 * Wd).sum() + ld)
    vy = np.var(yv)
    o = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(o.x); sw = np.sqrt(1 / (sg * ev + se))
    yw = (U.T @ yv) * sw; Xw = (U.T @ X) * sw[:, None]
    Q, _ = np.linalg.qr(Xw)
    yt = yw - Q @ (Q.T @ yw)
    Gw = (U.T @ Gs.T) * sw[:, None]
    Gt = Gw - Q @ (Q.T @ Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    b = (Gt.T @ yt) / gg
    df = len(yv) - X.shape[1] - 1
    sse = (yt @ yt) - b ** 2 * gg
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b / np.sqrt(sse / df / gg)
    p = np.full(Gm.shape[0], np.nan); ok = np.isfinite(t)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), df)
    return p, sg, se, ev, m, Ks

if MODE == "full":
    specs = (("no_PC", None),
             ("3_PC_centred", make_pc(False)),
             ("3_PC_standardised", make_pc(True)))
else:
    specs = (("no_PC", None),
             ("3_PC_standardised", make_pc(True)))

rows = []
for lab, PC in specs:
    print(f"\n### {lab}")
    for tr in TRAITS:
        y = mean_ph[tr].reindex(fathers).values.astype(float)
        p, sg, se, ev, m, Ks = emmax(y, Gi, K, PC)
        vi = np.isfinite(p)
        lam = np.median(stats.chi2.isf(p[vi], 1)) / stats.chi2.ppf(0.5, 1)
        n5e8 = int((p[vi] < 5e-8).sum())
        A = sg * Ks + se * np.eye(m.sum())
        Lc = np.linalg.cholesky(A + 1e-9 * np.eye(m.sum()))
        mu = np.nanmean(y[m]); rng = np.random.default_rng(SEED)
        mins = []
        for _ in range(NBOOT):
            ys = mu + Lc @ rng.standard_normal(m.sum())
            yf = np.full(len(fathers), np.nan); yf[m] = ys
            pb, *_ = emmax(yf, Gi, K, PC)
            mins.append(np.nanmin(pb))
        if MODE == "freeze":
            np.save(HERE / f"minp_{PANEL}_{lab.replace(chr(32),chr(95)).replace(chr(40),chr(95)).replace(chr(41),chr(95))}_{tr}.npy", np.array(mins))
        thr = np.quantile(mins, 0.05)
        nemp = int((p[vi] < thr).sum())
        rows.append({"panel": PANEL, "covariates": lab, "trait": tr, "λ": round(lam, 2),
                     "n_5e-8": n5e8, "emp_threshold": thr, "n_emp": nemp})
        print(f"  {tr:14s} λ={lam:.2f}  5e-8: {n5e8:4d}   emp.thr={thr:.1e}: {nemp:4d}")

df = pd.DataFrame(rows)
out_name = f"pca_{MODE}_{PANEL}.csv"
df.to_csv(HERE / out_name, index=False, encoding="utf-8-sig")
print(f"\n=== summary, {PANEL} lines ===")
print(df.pivot(index="trait", columns="covariates", values=["n_5e-8","n_emp"]).to_string())
print(f"\nwrote {out_name}")
