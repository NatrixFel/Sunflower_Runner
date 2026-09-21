"""
Uncertainty of the counters at the EMPIRICAL threshold.

The empirical threshold is a quantile of a Monte Carlo distribution, i.e. itself
an estimate with its own error. That error is measured here: a bootstrap interval
for the threshold is built from the saved minimum-p distributions (`minp_*.npy`,
2000 replicates, seed 20260817), and the range of significant-SNP counts is reported.

Observed p-values are recomputed deterministically (no bootstrap is needed for them).
Output: 261_threshold_uncertainty.csv
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
from scipy import stats, optimize
import warnings, glob
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
HERE = OUT
MINP = HERE / "minp"          # directory of saved distributions
NPZ = DEPOSIT / "23_merged_genotypes.npz"
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
CONF = ["LI29", "LI30"]
TRAITS = ["oil_content", "hull_content", "seed_weight_1000"]

z = np.load(NPZ, allow_pickle=True)
nalt = np.where(z["genotypes"].astype(float) < 0, np.nan, z["genotypes"].astype(float))
chrn = np.array([CHRMAP.get(c, -1) for c in z["chrom"].astype(str)])
samples = en_ids(z["samples"].astype(str))
ALLF = [s for s in samples if is_line_id(s)]
ph = en_table(pd.read_parquet(OUT / "221_lines_tidy.parquet"))
mean_ph = ph.groupby(["genotype", "trait"])["value"].mean().unstack()

def emmax(y, Gm, Kk, PC):
    m = ~np.isnan(y); yv = y[m]; Ks = Kk[np.ix_(m, m)]; Gs = Gm[:, m]
    ev, U = np.linalg.eigh(Ks); ev = np.maximum(ev, 1e-9)
    X = np.ones((len(yv), 1)) if PC is None else np.column_stack([np.ones(len(yv)), PC[m]])
    yr = U.T @ yv; Xr = U.T @ X
    def nll(t):
        sg, se = np.exp(t); D = sg * ev + se; W = 1 / D
        XtWX = Xr.T @ (Xr * W[:, None])
        try: inv = np.linalg.inv(XtWX)
        except Exception: return 1e10
        b = inv @ (Xr.T @ (yr * W)); r = yr - Xr @ b
        _, ld = np.linalg.slogdet(XtWX)
        return 0.5 * (np.log(D).sum() + (r ** 2 * W).sum() + ld)
    vy = np.var(yv)
    o = optimize.minimize(nll, np.log([.5 * vy, .5 * vy]), method="Nelder-Mead")
    sg, se = np.exp(o.x); sw = np.sqrt(1 / (sg * ev + se))
    yw = (U.T @ yv) * sw; Xw = (U.T @ X) * sw[:, None]
    Q, _ = np.linalg.qr(Xw); yt = yw - Q @ (Q.T @ yw)
    Gw = (U.T @ Gs.T) * sw[:, None]; Gt = Gw - Q @ (Q.T @ Gw)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    b = (Gt.T @ yt) / gg; df = len(yv) - X.shape[1] - 1
    sse = (yt @ yt) - b ** 2 * gg; sse = np.where(sse > 1e-12, sse, np.nan)
    t = b / np.sqrt(sse / df / gg)
    p = np.full(Gm.shape[0], np.nan); ok = np.isfinite(t)
    p[ok] = 2 * stats.t.sf(np.abs(t[ok]), df)
    return p

rng = np.random.default_rng(1)
rows = []
for tag, lines, pn in (("54_with_confectionery", ALLF, "54"),
                       ("52_oilseed_only", [f for f in ALLF if f not in CONF], "52")):
    col = [samples.index(l) for l in lines]
    G = nalt[:, col]
    cr = np.mean(~np.isnan(G), 1); af = np.nanmean(G, 1) / 2
    keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (np.minimum(af, 1 - af) >= 0.05)
    Gi = G[keep]; Gi = np.where(np.isnan(Gi), np.nanmean(Gi, axis=1, keepdims=True), Gi)
    p_al = Gi.mean(1) / 2; W = Gi - 2 * p_al[:, None]
    K = (W.T @ W) / (2 * np.sum(p_al * (1 - p_al)))
    X0 = Gi.T.astype(float)
    Xc = (X0 - X0.mean(0)) / (X0.std(0) + 1e-9)
    Us, Ss, _ = np.linalg.svd(Xc, full_matrices=False)
    PC3 = Us[:, :3] * Ss[:3]
    for model, PC, key in (("no_PC", None, "no_PC"),
                           ("3_PC_canon", PC3, "3_PC_standardised")):
        for tr in TRAITS:
            f = MINP / f"minp_{pn}_{key}_{tr}.npy"
            if not f.exists():
                cand = list(MINP.glob(f"minp_{pn}_*{tr}.npy"))
                cand = [c for c in cand if ("no_PC" in c.name) == (PC is None)]
                if not cand: continue
                f = cand[0]
            mp = np.load(f)
            thr = np.quantile(mp, 0.05)
            bs = np.array([np.quantile(rng.choice(mp, len(mp), replace=True), 0.05)
                           for _ in range(2000)])
            lo, hi = np.quantile(bs, [0.025, 0.975])
            y = mean_ph[tr].reindex(lines).values.astype(float)
            p = emmax(y, Gi, K, PC)
            vi = np.isfinite(p)
            n, n_lo, n_hi = (int((p[vi] < t).sum()) for t in (thr, lo, hi))
            rows.append({"panel": tag, "model": model, "trait": tr,
                         "threshold": thr, "threshold_CI_lo": lo, "threshold_CI_hi": hi,
                         "n": n, "n_at_CI_lo": n_lo, "n_at_CI_hi": n_hi})
            print(f"  {tag[:2]} {model:14s} {tr:14s} threshold={thr:.2e} "
                  f"[{lo:.2e}; {hi:.2e}]   n={n:3d}  range {n_lo}–{n_hi}")

df = pd.DataFrame(rows)
df.to_csv(HERE / "261_threshold_uncertainty.csv", index=False, encoding="utf-8-sig")
print("\n=== SUMS BY MODEL (three affected traits) ===")
for (tag, model), g in df.groupby(["panel", "model"]):
    print(f"  {tag:22s} {model:14s} n={g['n'].sum():4d}   "
          f"range {g['n_at_CI_lo'].sum()}–{g['n_at_CI_hi'].sum()}")
print("\nWritten: 261_threshold_uncertainty.csv")
