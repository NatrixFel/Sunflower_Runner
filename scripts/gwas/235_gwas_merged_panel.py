# -*- coding: utf-8 -*-
"""G1–G2. Panels from the merged set, PCA, K, EMMAX and BLINK, key loci.
The manuscript is not edited, computation only. Results are written to CSV."""
import sys
from pathlib import Path as _PathBoot
_p = _PathBoot(__file__).resolve().parent
while _p != _p.parent and not (_p / "paths.py").exists():
    _p = _p.parent
if str(_p) not in sys.path:
    sys.path.insert(0, str(_p))
from paths import DEPOSIT, WORK, CALLING, OUT, INTERMEDIATE, MODELS, use_models, work, calling, en_table, en_ids, en_id, is_line_id
use_models()
import numpy as np, pandas as pd, pathlib, sys, time, json
from scipy import stats, optimize
import warnings; warnings.filterwarnings("ignore")

ROOT = DEPOSIT
D = DEPOSIT
SCR = OUT
from blink import blink

CONF = {"LI29", "LI30"}
TRAITS = ["oil_content", "hull_content", "seed_weight_1000", "plant_height", "head_diameter",
          "days_emergence_flowering", "days_central_to_side", "autofertility_self", "autofertility_open"]

# ---------------------------------------------------------------- data
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
G = z["genotypes"].astype(float); G[G < 0] = np.nan
samp = en_ids(z["samples"].astype(str))
chrom_raw = [str(c) for c in z["chrom"]]
def cn(c):
    try: return int(c.replace("CM00", "").replace(".2", "")) - 7889
    except Exception: return -1
chrom = np.array([cn(c) for c in chrom_raw])
pos = np.asarray(z["pos"]).astype(np.int64)

lines54 = [s for s in samp if is_line_id(s)]
lines52 = [s for s in lines54 if s not in CONF]

ph = en_table(pd.read_parquet(OUT / "221_lines_tidy.parquet"))
def pheno(lines):
    out = {}
    for tr in TRAITS:
        s = ph[ph["trait"] == tr].groupby("genotype")["value"].mean()
        out[tr] = np.array([s.get(l, np.nan) for l in lines])
    return out

KEY_LOCI = [("chr8:782740", 8, 782740), ("chr17:50245155", 17, 50245155),
            ("chr17:50459845", 17, 50459845), ("chr5:~90.15M", 5, 90150000),
            ("chr7:~144.16M", 7, 144160000)]

# ---------------------------------------------------------------- tools
def vanraden(Gm):
    p = Gm.mean(1) / 2
    Zc = Gm - 2 * p[:, None]
    return (Zc.T @ Zc) / (2 * np.sum(p * (1 - p)))

def pca(Gm, standardise=True):
    A = Gm.T.astype(float); A = A - A.mean(0, keepdims=True)
    if standardise:
        sd = A.std(0, ddof=0); sd[sd < 1e-12] = 1.0; A = A / sd
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    return U * S, S ** 2 / np.sum(S ** 2)

def reml_vc(y, U, eig, X):
    yr = U.T @ y; Xr = U.T @ X; n, pn = len(y), X.shape[1]
    def nll(th):
        s2g, s2e = np.exp(th)
        w = 1.0 / (s2g * eig + s2e)
        Xw = Xr * np.sqrt(w)[:, None]; yw = yr * np.sqrt(w)
        b, *_ = np.linalg.lstsq(Xw, yw, rcond=None); r = yw - Xw @ b
        _, ldx = np.linalg.slogdet(Xw.T @ Xw)
        return 0.5 * (-np.sum(np.log(w)) + ldx + (n - pn) * np.log(max(r @ r, 1e-300)))
    v = np.var(y); best = None
    for st in ([0.5*v, 0.5*v], [0.9*v, 0.1*v], [0.1*v, 0.9*v]):
        r = optimize.minimize(nll, np.log(st), method="Nelder-Mead",
                              options=dict(maxiter=600, xatol=1e-7, fatol=1e-7))
        if best is None or r.fun < best.fun: best = r
    return np.exp(best.x)

def emmax(y, K, Gm, PCs):
    ok = ~np.isnan(y)
    y = y[ok]; Ks = K[np.ix_(ok, ok)]; Gs = Gm[:, ok]
    X = np.ones((len(y), 1))
    if PCs is not None: X = np.column_stack([X, PCs[ok]])
    eig, U = np.linalg.eigh(Ks); eig = np.maximum(eig, 1e-9)
    s2g, s2e = reml_vc(y, U, eig, X)
    w = np.sqrt(1.0 / (s2g * eig + s2e)); w = w / w.mean()
    yw = (U.T @ y) * w; Xw = (U.T @ X) * w[:, None]
    Q, _ = np.linalg.qr(Xw)
    yr = yw - Q @ (Q.T @ yw)
    Gw = (Gs @ U) * w[None, :]
    Gr = Gw - (Gw @ Q) @ Q.T
    num = Gr @ yr; den = np.einsum("ij,ij->i", Gr, Gr)
    okd = den > 1e-12 * max(den.max(), 1e-300)
    beta = np.zeros(len(den)); beta[okd] = num[okd] / den[okd]
    rss = (yr @ yr) - beta ** 2 * den
    df = len(y) - X.shape[1] - 1
    se = np.sqrt(np.maximum(rss, 1e-300) / df / np.maximum(den, 1e-300))
    t = np.where(okd & (se > 0), beta / se, 0.0)
    pv = np.full(len(den), np.nan); pv[okd] = 2 * stats.t.sf(np.abs(t[okd]), df)
    v = ~np.isnan(pv)
    lam = np.median(stats.chi2.isf(pv[v], 1)) / stats.chi2.ppf(0.5, 1) if v.sum() else np.nan
    return pv, lam

# ---------------------------------------------------------------- panels
def build_panel(lines, thr, full=False):
    idx = [samp.index(l) for l in lines]
    Gl = G[:, idx]
    cr = np.mean(~np.isnan(Gl), 1)
    ch_ok = (chrom >= 1) & (chrom <= 17)
    step1 = ch_ok & ((cr == 1.0) if full else (cr >= thr))
    af = np.nanmean(Gl, 1) / 2; maf = np.minimum(af, 1 - af)
    step2 = step1 & (maf >= 0.05)
    Gp = Gl[step2]
    miss = float(np.isnan(Gp).mean())
    Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
    return dict(mask=step2, G=Gi, n_start=len(cr), drop_chr=int(np.sum(~ch_ok)),
                drop_cr=int(np.sum(ch_ok) - np.sum(step1)),
                drop_maf=int(np.sum(step1) - np.sum(step2)),
                m=int(step2.sum()), miss=miss, lines=lines)

PANELS = [("call rate ≥ 0.9", 0.9, False), ("call rate ≥ 0.8", 0.8, False),
          ("call rate ≥ 0.5", 0.5, False), ("full_coverage", 1.0, True)]

rows_panel, rows_gwas, rows_loci = [], [], []
RUN_BLINK = "--blink" in sys.argv
for pname, thr, full in PANELS:
    for lset, lname in ((lines54, "54_lines"), (lines52, "52_lines")):
        t0 = time.time()
        P = build_panel(lset, thr, full)
        Gi = P["G"]; m = P["m"]
        PCs, var = pca(Gi, True)
        conf = np.array([1.0 if l in CONF else 0.0 for l in lset])
        r_pc1 = abs(np.corrcoef(PCs[:, 0], conf)[0, 1]) if conf.std() > 0 else np.nan
        K = vanraden(Gi)
        rows_panel.append(dict(panel=pname, n_lines=lname, n_markers=m,
                               dropped_cr=P["drop_cr"], dropped_maf=P["drop_maf"],
                               missingness=P["miss"], PC1=var[0]*100, PC2=var[1]*100,
                               PC3=var[2]*100, corr_PC1_type=r_pc1,
                               bonferroni=0.05/m))
        phen = pheno(lset)
        sub_chrom = chrom[P["mask"]]; sub_pos = pos[P["mask"]]
        for tr in TRAITS:
            y = phen[tr]
            if np.sum(~np.isnan(y)) < 20: continue
            pv, lam = emmax(y, K, Gi, PCs[:, :3])
            n5 = int(np.nansum(pv < 5e-8)); nb = int(np.nansum(pv < 0.05 / m))
            rec = dict(panel=pname, n_lines=lname, trait=tr, model="EMMAX",
                       n_markers=m, lam=lam, n_sig_5e8=n5, n_sig_bonf=nb,
                       min_p=float(np.nanmin(pv)))
            rows_gwas.append(rec)
            for nm, ch, po in KEY_LOCI:
                hit = np.where((sub_chrom == ch) & (np.abs(sub_pos - po) <= (200000 if "~" in nm else 0)))[0]
                rows_loci.append(dict(panel=pname, n_lines=lname, trait=tr, locus=nm,
                                      present=len(hit) > 0,
                                      p=float(np.nanmin(pv[hit])) if len(hit) else np.nan,
                                      markers_in_window=len(hit)))
            if RUN_BLINK:
                try:
                    res = blink(y, Gi, sub_chrom, sub_pos, PCs[:, :3])
                    pb = res["pvals"]
                    rows_gwas.append(dict(panel=pname, n_lines=lname, trait=tr, model="BLINK",
                                          n_markers=m, lam=np.nan,
                                          n_sig_5e8=int(np.nansum(pb < 5e-8)),
                                          n_sig_bonf=int(np.nansum(pb < 0.05 / m)),
                                          min_p=float(np.nanmin(pb))))
                except Exception as e:
                    rows_gwas.append(dict(panel=pname, n_lines=lname, trait=tr, model="BLINK",
                                          n_markers=m, lam=np.nan, n_sig_5e8=-1, n_sig_bonf=-1,
                                          min_p=np.nan))
        print(f"{pname:18s} {lname} m={m:6d} miss={P['miss']*100:5.2f}% "
              f"PC1={var[0]*100:5.2f}% corr={r_pc1 if not np.isnan(r_pc1) else float('nan'):.3f} "
              f"| {time.time()-t0:.0f} s", flush=True)

pd.DataFrame(rows_panel).to_csv(SCR / "235_gwas_merged_panel.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_gwas).to_csv(SCR / "gmp_gwas.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_loci).to_csv(SCR / "gmp_loci.csv", index=False, encoding="utf-8-sig")
print("\nwritten to", SCR)
