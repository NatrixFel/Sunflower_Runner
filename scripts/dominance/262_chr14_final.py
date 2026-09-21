"""
chr14:169.21M — FINAL check.

Model:  y ~ 1 + mother(2 df) + SNP_dose + SNP_het,
         u_a ~ N(0, s2a*Ka), u_d ~ N(0, s2d*Kd), e ~ N(0, s2e*I)

Two null distributions of the genome-wide minimum p (both with RE-ESTIMATED
variance components on every replicate — the step that was done incorrectly
in the previous run):
  A) PARAMETRIC BOOTSTRAP — y is simulated under the fitted null
     model, so the polygenic covariance is PRESERVED. Primary test.
  B) PHENOTYPE PERMUTATION — the covariance is broken. Control test.

Variance components are parameterized as fractions (ha, hd); the overall scale
is profiled analytically and does not affect the t-statistics.
"""

import argparse
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
from scipy import stats, optimize
from pathlib import Path
import time, warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--trait", default="seed_weight_1000")
parser.add_argument("--phenotypes", default="222_hybrid_blup_phenotypes.csv",
                    help="outputs/ CSV of hybrid BLUP phenotypes")
args = parser.parse_args()

ROOT = DEPOSIT
NPZ = DEPOSIT / "23_merged_genotypes.npz"
HERE = OUT
CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
CHRMAP = {f"CM00{7889+i}.2": i for i in range(1, 18)}
TRAIT = args.trait
NREP = 500

# ---------------- data ----------------
z = np.load(NPZ, allow_pickle=True)
disc = z["genotypes"].astype(float); chrom = z["chrom"].astype(str)
pos = z["pos"]; samples = en_ids(z["samples"].astype(str))
nalt = np.where(disc < 0, np.nan, disc)
chrn = np.array([CHRMAP.get(c, -1) for c in chrom])
cr = np.mean(~np.isnan(nalt), 1); af0 = np.nanmean(nalt, 1) / 2
maf = np.minimum(af0, 1 - af0)
keep = (chrn >= 1) & (chrn <= 17) & (cr >= 0.9) & (maf >= 0.05)
gt = nalt[keep]; gt = np.where(np.isnan(gt), np.nanmean(gt, axis=1, keepdims=True), gt)
chf = chrn[keep]; pf = pos[keep]; n_snp = gt.shape[0]
s2c = {s: i for i, s in enumerate(samples)}

bl = en_table(pd.read_csv(OUT / args.phenotypes))
bl[["mom_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)
bl = bl[bl["mom"].isin(set(samples)) & bl["father"].isin(set(samples))]
d = bl.dropna(subset=[TRAIT]).copy()
d = d[~d["father"].isin(CONF)].reset_index(drop=True)
y = d[TRAIT].values.astype(float); n = len(d)

HG = np.empty((n, n_snp))
for i, row in d.iterrows():
    HG[i] = (gt[:, s2c[row["mom"]]] + gt[:, s2c[row["father"]]]) / 2
IH = (np.abs(HG - 1) < 0.25).astype(float)
p_al = HG.mean(0) / 2
Wa = HG - 2 * p_al
Ka = (Wa @ Wa.T) / (2 * np.sum(p_al * (1 - p_al)))
exp_het = 2 * p_al * (1 - p_al)
Wd = IH - exp_het
Kd = (Wd @ Wd.T) / np.sum(exp_het * (1 - exp_het))
X = np.column_stack([np.ones(n), (d["mom"] == "VA761").astype(float),
                     (d["mom"] == "VK934").astype(float)])
useful = (IH.sum(0) >= 5) & (IH.std(0) > 1e-6) & (HG.std(0) > 1e-6)
del nalt, disc, gt, Wa, Wd
nX = X.shape[1]; df = n - nX - 2
print(f"Panel {n_snp:,} SNP; n={n}; testable {int(useful.sum()):,}", flush=True)

I_n = np.eye(n)

# ---------------- REML over fractions (ha, hd) ----------------
def reml_prof(h, yv):
    ha, hd = h
    if ha < 0 or hd < 0 or ha + hd > 0.999: return 1e10
    A = ha * Ka + hd * Kd + (1 - ha - hd) * I_n
    try: L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError: return 1e10
    Ai_X = np.linalg.solve(A, X); XtAX = X.T @ Ai_X
    try: b = np.linalg.solve(XtAX, Ai_X.T @ yv)
    except np.linalg.LinAlgError: return 1e10
    r = yv - X @ b
    q = r @ np.linalg.solve(A, r)
    if q <= 0: return 1e10
    ldA = 2 * np.sum(np.log(np.diag(L)))
    _, ldX = np.linalg.slogdet(XtAX)
    return 0.5 * (ldA + ldX + (n - nX) * np.log(q / (n - nX)))

def fit_vc(yv, starts=((.5, .2), (.2, .5), (.05, .05))):
    best = None
    for st in starts:
        r = optimize.minimize(reml_prof, np.array(st), args=(yv,), method="Nelder-Mead",
                              options={"maxiter": 400, "fatol": 1e-7, "xatol": 1e-5})
        if best is None or r.fun < best.fun: best = r
    ha, hd = np.clip(best.x, 0, None)
    if ha + hd > 0.999: ha, hd = ha / (ha + hd) * .999, hd / (ha + hd) * .999
    return ha, hd

def scan(yv, ha, hd):
    """Genome-wide scan of p_dom at given variance fractions."""
    A = ha * Ka + hd * Kd + (1 - ha - hd) * I_n
    ev, U = np.linalg.eigh(A); ev = np.maximum(ev, 1e-10)
    sw = 1 / np.sqrt(ev)
    Xw = (U.T @ X) * sw[:, None]
    Q, _ = np.linalg.qr(Xw)
    yw = (U.T @ yv) * sw; yt = yw - Q @ (Q.T @ yw)
    Gt = (U.T @ HG) * sw[:, None]; Gt -= Q @ (Q.T @ Gt)
    Ht = (U.T @ IH) * sw[:, None]; Ht -= Q @ (Q.T @ Ht)
    gg = np.einsum('ij,ij->j', Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    gh = np.einsum('ij,ij->j', Gt, Ht)
    gy = Gt.T @ yt
    Hp = Ht - Gt * (gh / gg)
    del Gt, Ht
    hh = np.einsum('ij,ij->j', Hp, Hp)
    hy = Hp.T @ yt
    ok = useful & np.isfinite(hh) & (hh > 1e-10) & np.isfinite(gg)
    b2 = hy / np.where(ok, hh, np.nan)
    sse = (yt @ yt) - (gy ** 2) / gg - b2 ** 2 * hh
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b2 / np.sqrt(sse / df / hh)
    p = np.full(n_snp, np.nan)
    m = ok & np.isfinite(t)
    p[m] = 2 * stats.t.sf(np.abs(t[m]), df)
    return p, b2

# ---------------- observed ----------------
t0 = time.time()
ha0, hd0 = fit_vc(y)
print(f"REML on real data: Ka fraction={ha0:.3f}, Kd={hd0:.3f}, residual={1-ha0-hd0:.3f}", flush=True)
obs, beta = scan(y, ha0, hd0)
vi = np.isfinite(obs)
lam = np.median(stats.chi2.isf(obs[vi], 1)) / stats.chi2.ppf(0.5, 1)
bi = int(np.nanargmin(np.where(vi, obs, np.inf)))
print(f"scan in {time.time()-t0:.1f} s;  lambda={lam:.2f};  tested {int(vi.sum()):,}")
print(f"BEST: chr{int(chf[bi])}:{int(pf[bi])}  p_dom={obs[bi]:.3e}  beta={beta[bi]:+.3f}")
for j in np.argsort(np.where(vi, obs, np.inf))[:8]:
    print(f"   chr{int(chf[j]):>2}:{int(pf[j]):>10}  p={obs[j]:.2e}  beta={beta[j]:+.2f}")

# ---------------- null distributions ----------------
A0 = ha0 * Ka + hd0 * Kd + (1 - ha0 - hd0) * I_n
Lc = np.linalg.cholesky(A0 + 1e-10 * I_n)
Ai_X = np.linalg.solve(A0, X)
b0 = np.linalg.solve(X.T @ Ai_X, Ai_X.T @ y)
r0 = y - X @ b0
s2_hat = (r0 @ np.linalg.solve(A0, r0)) / (n - nX)
print(f"\nnull model: b={np.round(b0,2)}, s2={s2_hat:.2f}", flush=True)

rng = np.random.default_rng(20260816)
res = {}
for mode in ("bootstrap", "permutation"):
    minp = np.empty(NREP); t0 = time.time()
    for k in range(NREP):
        if mode == "bootstrap":
            yv = X @ b0 + np.sqrt(s2_hat) * (Lc @ rng.standard_normal(n))
        else:
            yv = y[rng.permutation(n)]
        ha, hd = fit_vc(yv, starts=((ha0, hd0), (.05, .05)))
        pp, _ = scan(yv, ha, hd)
        minp[k] = np.nanmin(pp)
        if (k + 1) % 100 == 0:
            print(f"  [{mode}] {k+1}/{NREP} ({time.time()-t0:.0f} s) "
                  f"thr5%={np.quantile(minp[:k+1],0.05):.2e}", flush=True)
    res[mode] = minp
    tag = "" if TRAIT == "seed_weight_1000" else f"_{TRAIT}"
    pd.DataFrame({"min_p": minp}).to_csv(HERE / f"262_null_{mode}{tag}.csv", index=False)

print(f"\n{'='*72}\nRESULT chr14 (Ka+Kd model with mother effects, {NREP} replicates, {TRAIT})")
print(f"  observed best p_dom = {obs[bi]:.3e} @ chr{int(chf[bi])}:{int(pf[bi])}")
for mode in ("bootstrap", "permutation"):
    m = res[mode]
    pe = (np.sum(m <= obs[bi]) + 1) / (NREP + 1)
    print(f"  [{mode:11s}] thr5%={np.quantile(m,0.05):.2e}  thr1%={np.quantile(m,0.01):.2e}  "
          f"genome-wide p={pe:.4f}  -> {'SIGNIFICANT' if pe<0.05 else 'NOT SIGNIFICANT'}")
tag = "" if TRAIT == "seed_weight_1000" else f"_{TRAIT}"
pd.DataFrame({"chr": chf, "pos": pf, "p_dom": obs, "beta_dom": beta}).to_csv(
    HERE / f"262_obs_scan_final{tag}.csv", index=False)
print(f"\nSaved: 262_obs_scan_final{tag}.csv, 262_null_bootstrap{tag}.csv, 262_null_permutation{tag}.csv")
