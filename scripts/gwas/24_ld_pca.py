# -*- coding: utf-8 -*-
"""P1. LD pruning, PCA on pruned sets, recomputation of N-50, N-51 and λ.
The manuscript is not edited. Computation only.

  1. analysis panel of 54 lines (Chr1–17, call rate >= 0.9, MAF >= 0.05) — control 81 903 SNPs;
  2. sliding-window LD pruning within chromosome at r2 = 0.2 / 0.5 / 0.9;
  3. PCA on the pruned set, two preprocessing variants: centering+standardization
     (the declared procedure) and centering only;
  4. corr(PC1, confectionery-type indicator)  -> N-50;
  5. EMMAX on the FULL analysis panel with 3 PCs from the pruned set -> SNP count at p < 5e-8
     (N-51) and λ. K_a is computed on the full panel in all variants.
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
import numpy as np, pandas as pd, pathlib, sys, time
from scipy import stats, optimize
import warnings; warnings.filterwarnings("ignore")

ROOT = DEPOSIT
D = DEPOSIT
SCRATCH = OUT
CONF = {"LI29", "LI30"}
TRAITS = ["oil_content", "hull_content", "seed_weight_1000"]

# ---------------- 1. analysis panel ----------------
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
def cn(c):
    try: return int(str(c).replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1
chrom = np.array([cn(c) for c in z["chrom"]])
pos = np.asarray(z["pos"]).astype(np.int64)
samp = en_ids(z["samples"].astype(str))
lines = [s for s in samp if is_line_id(s)]
idx = [samp.index(s) for s in lines]
Gl = G[:, idx]
cr = np.mean(~np.isnan(Gl), 1); af = np.nanmean(Gl, 1) / 2; maf = np.minimum(af, 1 - af)
keep = (chrom >= 1) & (chrom <= 17) & (cr >= 0.9) & (maf >= 0.05)
Gp = Gl[keep]; chrom_p = chrom[keep]; pos_p = pos[keep]
Gp = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
print(f"analysis panel of 54 lines: {Gp.shape[0]} SNP (81 903 in the paper), samples {Gp.shape[1]}")
conf_ind = np.array([1.0 if l in CONF else 0.0 for l in lines])

# ---------------- 2. LD pruning ----------------
def ld_prune(Gm, chroms, positions, r2_thr, win=50, step=5):
    keep_mask = np.ones(Gm.shape[0], dtype=bool)
    for ch in np.unique(chroms):
        ii = np.where(chroms == ch)[0]
        ii = ii[np.argsort(positions[ii])]
        start = 0
        while start < len(ii):
            w = ii[start:start + win]
            w = w[keep_mask[w]]
            if len(w) > 1:
                X = Gm[w]
                Xc = X - X.mean(1, keepdims=True)
                sd = Xc.std(1); sd[sd < 1e-12] = np.inf
                Z = Xc / sd[:, None]
                R2 = ((Z @ Z.T) / Xc.shape[1]) ** 2
                for a in range(len(w)):
                    if not keep_mask[w[a]]: continue
                    for b in range(a + 1, len(w)):
                        if not keep_mask[w[b]]: continue
                        if R2[a, b] > r2_thr: keep_mask[w[b]] = False
            start += step
    return keep_mask

# ---------------- 3. PCA ----------------
def pca(Gm, standardise):
    X = Gm.T.astype(float)
    X = X - X.mean(0, keepdims=True)
    if standardise:
        sd = X.std(0, ddof=0); sd[sd < 1e-12] = 1.0
        X = X / sd
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    var = S ** 2 / np.sum(S ** 2)
    return U * S, var

# ---------------- 4. EMMAX ----------------
def vanraden(Gm):
    p = Gm.mean(1) / 2
    Zc = Gm - 2 * p[:, None]
    return (Zc.T @ Zc) / (2 * np.sum(p * (1 - p)))

def reml_vc(y, U, eig, X):
    yr = U.T @ y; Xr = U.T @ X
    n, pn = len(y), X.shape[1]
    def nll(th):
        s2g, s2e = np.exp(th)
        w = 1.0 / (s2g * eig + s2e)
        Xw = Xr * np.sqrt(w)[:, None]; yw = yr * np.sqrt(w)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
        r = yw - Xw @ b
        _, ldx = np.linalg.slogdet(Xw.T @ Xw)
        return 0.5 * (-np.sum(np.log(w)) + ldx + (n - pn) * np.log(max(r @ r, 1e-300)))
    v = np.var(y); best = None
    for st in ([0.5*v, 0.5*v], [0.9*v, 0.1*v], [0.1*v, 0.9*v]):
        r = optimize.minimize(nll, np.log(st), method="Nelder-Mead",
                              options=dict(maxiter=800, xatol=1e-7, fatol=1e-7))
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x)

def emmax_fast(y, K, Gm, PCs):
    m_ok = ~np.isnan(y)
    y = y[m_ok]; Ks = K[np.ix_(m_ok, m_ok)]; Gs = Gm[:, m_ok]
    X = np.ones((len(y), 1))
    if PCs is not None: X = np.column_stack([X, PCs[m_ok]])
    eig, U = np.linalg.eigh(Ks); eig = np.maximum(eig, 1e-9)
    s2g, s2e = reml_vc(y, U, eig, X)
    w = np.sqrt(1.0 / (s2g * eig + s2e))
    w = w / w.mean()          # scale does not affect the t-statistic, but prevents denormalization
    yw = (U.T @ y) * w
    Xw = (U.T @ X) * w[:, None]
    Q, _ = np.linalg.qr(Xw)
    yr = yw - Q @ (Q.T @ yw)
    Gw = (Gs @ U) * w[None, :]
    Gr = Gw - (Gw @ Q) @ Q.T
    num = Gr @ yr
    den = np.einsum("ij,ij->i", Gr, Gr)
    ok = den > 1e-12 * max(den.max(), 1e-300)
    beta = np.zeros(len(den)); beta[ok] = num[ok] / den[ok]
    rss = (yr @ yr) - beta ** 2 * den
    df = len(y) - X.shape[1] - 1
    se = np.sqrt(np.maximum(rss, 1e-300) / df / np.maximum(den, 1e-300))
    tst = np.where(ok & (se > 0), beta / se, 0.0)
    pv = np.full(len(den), np.nan)
    pv[ok] = 2 * stats.t.sf(np.abs(tst[ok]), df)
    v = ~np.isnan(pv)
    lam = np.median(stats.chi2.isf(pv[v], 1)) / stats.chi2.ppf(0.5, 1) if v.sum() else np.nan
    return pv, lam

# ---------------- phenotypes ----------------
ph = en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
phen = {}
for tr in TRAITS:
    s = ph[ph["trait"] == tr].groupby("genotype")["value"].mean()
    phen[tr] = np.array([s.get(l, np.nan) for l in lines])
    print(f"  phenotype {tr:14s}: non-missing {int(np.sum(~np.isnan(phen[tr])))} of {len(lines)}")

K = vanraden(Gp)

# ---------------- implementation check ----------------
from emmax import emmax_run
PC_std_full, var_std_full = pca(Gp, True)
sub = Gp[:3000]
res = emmax_run(phen["oil_content"], K, sub, PC_std_full[:, :3])
pv_ref, lam_ref = res[0], res[2]
pv_my, lam_my = emmax_fast(phen["oil_content"], K, sub, PC_std_full[:, :3])
ok = ~np.isnan(pv_ref) & ~np.isnan(pv_my)
print(f"\nEMMAX CHECK: max|log10 p_ref - log10 p_my| = "
      f"{np.max(np.abs(np.log10(pv_ref[ok]) - np.log10(pv_my[ok]))):.2e} on {int(ok.sum())} markers; "
      f"lam_ref={lam_ref:.4f} lam_my={lam_my:.4f}")

# ---------------- run ----------------
VARIANTS = [("full_set_as_published", None)] + [(f"LD r2 > {t}", t) for t in (0.2, 0.5, 0.9)]
rows = []
for vname, thr in VARIANTS:
    t0 = time.time()
    km = np.ones(Gp.shape[0], dtype=bool) if thr is None else ld_prune(Gp, chrom_p, pos_p, thr)
    Gk = Gp[km]
    for std, pname in ((True, "standardised"), (False, "centred_only")):
        PCs, var = pca(Gk, std)
        r_pc1 = abs(np.corrcoef(PCs[:, 0], conf_ind)[0, 1])
        counts, lams = {}, {}
        for tr in TRAITS:
            pv, lam = emmax_fast(phen[tr], K, Gp, PCs[:, :3])
            counts[tr] = int(np.nansum(pv < 5e-8)); lams[tr] = float(lam)
        rows.append(dict(variant=vname, n_markers=int(km.sum()), preprocess=pname,
                         PC1=var[0]*100, PC2=var[1]*100, PC3=var[2]*100, corr_PC1=r_pc1,
                         **{f"SNP_{t}": counts[t] for t in TRAITS},
                         total=sum(counts.values()),
                         lam_min=min(lams.values()), lam_max=max(lams.values())))
        print(f"  {vname:28s} {int(km.sum()):6d} SNP | {pname:22s} | PC1 {var[0]*100:5.2f}% | "
              f"corr {r_pc1:.3f} | sig. {sum(counts.values()):4d} "
              f"({', '.join(f'{t}:{counts[t]}' for t in TRAITS)}) | lam {min(lams.values()):.2f}-{max(lams.values()):.2f}"
              f" | {time.time()-t0:.0f} s", flush=True)
df = pd.DataFrame(rows)
df.to_csv(SCRATCH / "p1_results.csv", index=False, encoding="utf-8-sig")
print("\n" + df.round(3).to_string(index=False))
