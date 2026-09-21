# -*- coding: utf-8 -*-
"""PART B. Dissection of the chr17 locus: dominance scan on all panels plus an LD window.

The dominance-scan model matches 262_crosspanel.py:
mother fixed effects + Ka + Kd, REML profile over two variance fractions,
and a test of beta_dom given the additive dose.

The original phase6c model is also reproduced (14_phase6c_overdominance.py):
additive K only, no mother fixed effects, phenotype = means.

The manuscript is not edited. External VCFs are opened read-only.
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
sys.stdout.reconfigure(encoding="utf-8")

ROOT = DEPOSIT
D = DEPOSIT
EXT = DEPOSIT
SCR = OUT
SCR.mkdir(parents=True, exist_ok=True)

CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
TRAIT = "seed_weight_1000"

# ------------------------------------------------ merged panel
z = np.load(D / "23_merged_genotypes.npz", allow_pickle=True)
Graw = z["genotypes"].astype(float); Graw[Graw < 0] = np.nan
samp = [str(s) for s in z["samples"]]
chrom = np.array([int(str(c).replace("CM00", "").replace(".2", "")) - 7889 for c in z["chrom"]])
pos = np.asarray(z["pos"]).astype(np.int64)

# ------------------------------------------------ phase-6 panel (external VCF, read-only)
def read_phase6():
    f = DEPOSIT / "23_no_imputation.vcf"
    names, ch, po, rows = None, [], [], []
    with open(f, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("##"): continue
            p = line.rstrip("\n").split("\t")
            if line.startswith("#CHROM"):
                names = p[9:]; continue
            try: c = int(p[0].replace("CM00", "").replace(".2", "")) - 7889
            except Exception: c = -1
            ch.append(c); po.append(int(p[1]))
            g = []
            for x in p[9:]:
                gt = x.split(":")[0]
                g.append({"0/0": 0.0, "0/1": 1.0, "1/0": 1.0, "1/1": 2.0}.get(gt, np.nan))
            rows.append(g)
    SEQ = {i: "LI%d" % i for i in range(1, 33)}
    SEQ.update({33: "LI34", 34: "LI35"}); SEQ.update({i: "LI%d" % (i + 2) for i in range(35, 53)})
    SEQ.update({53: "LI33", 54: "LI36", 55: "VA761", 56: "VK101", 57: "VK934"})
    nm = []
    for s in names:
        try: nm.append(SEQ.get(int(float(s)), str(s)))
        except Exception: nm.append(str(s))
    return np.array(rows), np.array(ch), np.array(po, np.int64), nm

# ------------------------------------------------ hybrid phenotypes
bl = en_table(pd.read_csv(OUT / "222_hybrid_blup_phenotypes.csv"))
bl[["mom_long", "father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)

hm = en_table(pd.read_csv(OUT / "221_hybrid_means_delivered.csv"))

WINDOWS = [("chr14:169.20-169.23", 14, 169_200_000, 169_230_000),
           ("chr17:50.20-50.50", 17, 50_200_000, 50_500_000)]
MARKERS = [("chr17:50245155", 17, 50245155), ("chr17:50459845", 17, 50459845),
           ("chr14:169211003", 14, 169211003)]

def dominance_scan(G, chf, pf, names, y_source="BLUP", mother_fixed=True, use_Kd=True):
    """G: (m, n_samples) dosages with imputation; names — sample names."""
    s2c = {s: i for i, s in enumerate(names)}
    if y_source == "BLUP":
        d = bl.dropna(subset=[TRAIT]).copy()
        d = d[~d["father"].isin(CONF)]
        d = d[d["mom"].isin(s2c) & d["father"].isin(s2c)].reset_index(drop=True)
        y = d[TRAIT].values.astype(float)
        moms = d["mom"].values
    else:  # means, as in phase6c
        d = hm.copy()
        d["mom"] = d["mother_long"].map(LTV) if "mother_long" in d.columns else d["mother"].map(LTV)
        col = "seed_weight_1000" if "seed_weight_1000" in d.columns else "seed_weight_1000"
        d = d.dropna(subset=[col])
        d = d[d["mom"].isin(s2c) & d["father"].isin(s2c)].reset_index(drop=True)
        y = d[col].values.astype(float)
        moms = d["mom"].values
    n = len(d); n_snp = G.shape[0]
    HG = np.empty((n, n_snp))
    for i in range(n):
        HG[i] = (G[:, s2c[moms[i]]] + G[:, s2c[d["father"].values[i]]]) / 2
    IH = (np.abs(HG - 1) < 0.25).astype(float)
    p_al = HG.mean(0) / 2
    Wa = HG - 2 * p_al; Ka = (Wa @ Wa.T) / (2 * np.sum(p_al * (1 - p_al)))
    eh = 2 * p_al * (1 - p_al)
    Wd = IH - eh; Kd = (Wd @ Wd.T) / np.sum(eh * (1 - eh))
    if mother_fixed:
        X = np.column_stack([np.ones(n), (moms == "VA761").astype(float), (moms == "VK934").astype(float)])
    else:
        X = np.ones((n, 1))
    useful = (IH.sum(0) >= 5) & (IH.std(0) > 1e-6) & (HG.std(0) > 1e-6)
    nX = X.shape[1]; df = n - nX - 2; I_n = np.eye(n)

    def reml_prof(h, yv):
        ha, hd = h
        if ha < 0 or hd < 0 or ha + hd > 0.999: return 1e10
        A = ha * Ka + (hd * Kd if use_Kd else 0.0) + (1 - ha - hd) * I_n
        try: L = np.linalg.cholesky(A)
        except np.linalg.LinAlgError: return 1e10
        AiX = np.linalg.solve(A, X); XtAX = X.T @ AiX
        try: b = np.linalg.solve(XtAX, AiX.T @ yv)
        except np.linalg.LinAlgError: return 1e10
        r = yv - X @ b; q = r @ np.linalg.solve(A, r)
        if q <= 0: return 1e10
        _, ldX = np.linalg.slogdet(XtAX)
        return 0.5 * (2 * np.sum(np.log(np.diag(L))) + ldX + (n - nX) * np.log(q / (n - nX)))

    best = None
    for st in ((.5, .2), (.2, .5), (.05, .05)):
        r = optimize.minimize(reml_prof, np.array(st), args=(y,), method="Nelder-Mead",
                              options={"maxiter": 400, "fatol": 1e-7, "xatol": 1e-5})
        if best is None or r.fun < best.fun: best = r
    ha, hd = np.clip(best.x, 0, None)
    if ha + hd > 0.999:
        s = (ha + hd) / 0.999; ha, hd = ha / s, hd / s
    if not use_Kd: hd = 0.0

    A = ha * Ka + hd * Kd + (1 - ha - hd) * I_n
    ev, U = np.linalg.eigh(A); ev = np.maximum(ev, 1e-10); sw = 1 / np.sqrt(ev)
    Xw = (U.T @ X) * sw[:, None]; Q, _ = np.linalg.qr(Xw)
    yw = (U.T @ y) * sw; yt = yw - Q @ (Q.T @ yw)
    Gt = (U.T @ HG) * sw[:, None]; Gt -= Q @ (Q.T @ Gt)
    Ht = (U.T @ IH) * sw[:, None]; Ht -= Q @ (Q.T @ Ht)
    gg = np.einsum("ij,ij->j", Gt, Gt); gg = np.where(gg < 1e-12, np.nan, gg)
    gh = np.einsum("ij,ij->j", Gt, Ht); gy = Gt.T @ yt
    Hp = Ht - Gt * (gh / gg)
    hh = np.einsum("ij,ij->j", Hp, Hp); hy = Hp.T @ yt
    ok = useful & np.isfinite(hh) & (hh > 1e-10) & np.isfinite(gg)
    b2 = hy / np.where(ok, hh, np.nan)
    sse = (yt @ yt) - (gy ** 2) / gg - b2 ** 2 * hh
    sse = np.where(sse > 1e-12, sse, np.nan)
    t = b2 / np.sqrt(sse / df / hh)
    p = np.full(n_snp, np.nan); mk = ok & np.isfinite(t)
    p[mk] = 2 * stats.t.sf(np.abs(t[mk]), df)
    vi = np.isfinite(p)
    lam = np.median(stats.chi2.isf(p[vi], 1)) / stats.chi2.ppf(0.5, 1) if vi.sum() else np.nan
    nhet = IH.sum(0)
    return dict(p=p, beta=b2, lam=lam, n=n, ha=ha, hd=hd, ntest=int(vi.sum()),
                nhet=nhet, chf=chf, pf=pf, m=n_snp)

def summarise(tag, R):
    p, chf, pf = R["p"], R["chf"], R["pf"]
    vi = np.isfinite(p)
    bi = int(np.nanargmin(np.where(vi, p, np.inf)))
    out = dict(panel=tag, n_hybrids=R["n"], n_markers=R["m"], n_tested=R["ntest"],
               frac_Ka=round(R["ha"], 3), frac_Kd=round(R["hd"], 3), lam=round(R["lam"], 3),
               best="chr%d:%d" % (chf[bi], pf[bi]), best_p=p[bi], best_beta=R["beta"][bi],
               bonferroni=0.05 / max(R["ntest"], 1))
    for nm, c, lo, hi in WINDOWS:
        w = np.where((chf == c) & (pf >= lo) & (pf <= hi) & vi)[0]
        if len(w):
            b = w[int(np.argmin(p[w]))]
            out["%s_n" % nm] = len(w); out["%s_p" % nm] = p[b]
            out["%s_beta" % nm] = R["beta"][b]; out["%s_top" % nm] = int(pf[b])
        else:
            out["%s_n" % nm] = 0; out["%s_p" % nm] = np.nan
            out["%s_beta" % nm] = np.nan; out["%s_top" % nm] = None
    for nm, c, po in MARKERS:
        i = np.where((chf == c) & (pf == po))[0]
        if len(i):
            out["%s_present" % nm] = True; out["%s_p" % nm] = p[i[0]]
            out["%s_beta" % nm] = R["beta"][i[0]]; out["%s_nhet" % nm] = float(R["nhet"][i[0]])
        else:
            out["%s_present" % nm] = False; out["%s_p" % nm] = np.nan
            out["%s_beta" % nm] = np.nan; out["%s_nhet" % nm] = np.nan
    top = np.argsort(np.where(vi, p, np.inf))[:10]
    out["top10"] = "; ".join("chr%d:%d p=%.2e b=%+.2f" % (chf[j], pf[j], p[j], R["beta"][j]) for j in top)
    return out

# ------------------------------------------------ merged panels (57 samples)
def merged_panel(thr, full=False):
    cr = np.mean(~np.isnan(Graw), 1)
    ch_ok = (chrom >= 1) & (chrom <= 17)
    s1 = ch_ok & ((cr == 1.0) if full else (cr >= thr))
    af = np.nanmean(Graw, 1) / 2; maf = np.minimum(af, 1 - af)
    s2 = s1 & (maf >= 0.05)
    Gp = Graw[s2]
    Gi = np.where(np.isnan(Gp), np.nanmean(Gp, 1, keepdims=True), Gp)
    return Gi, chrom[s2], pos[s2]

rows = []
for tag, thr, full in [("merged call rate >= 0.9", 0.9, False),
                       ("merged call rate >= 0.8", 0.8, False),
                       ("merged call rate >= 0.5", 0.5, False),
                       ("merged full coverage", 1.0, True)]:
    G, ch, pf = merged_panel(thr, full)
    t0 = time.time()
    R = dominance_scan(G, ch, pf, samp)
    rows.append(summarise(tag, R))
    print("%-26s m=%6d n=%3d lam=%.2f best=%s p=%.2e | %.0f s"
          % (tag, R["m"], R["n"], R["lam"], rows[-1]["best"], rows[-1]["best_p"], time.time() - t0),
          flush=True)

# ------------------------------------------------ phase-6 panel
G6, ch6, po6, nm6 = read_phase6()
print("phase-6 VCF: %d variants x %d samples, missing %.4f %%"
      % (G6.shape[0], G6.shape[1], np.isnan(G6).mean() * 100), flush=True)
af6 = np.nanmean(G6, 1) / 2; maf6 = np.minimum(af6, 1 - af6)
k6 = (ch6 >= 1) & (ch6 <= 17) & (maf6 >= 0.05)
G6f = G6[k6]; G6f = np.where(np.isnan(G6f), np.nanmean(G6f, 1, keepdims=True), G6f)
R6 = dominance_scan(G6f, ch6[k6], po6[k6], nm6)
rows.append(summarise("phase-6 (42 289, no imputation)", R6))
print("phase-6 BLUP model: best %s p=%.2e" % (rows[-1]["best"], rows[-1]["best_p"]), flush=True)

# original phase6c model: means, no mother fixed effects, no Kd
R6c = dominance_scan(G6f, ch6[k6], po6[k6], nm6, y_source="means", mother_fixed=False, use_Kd=False)
rows.append(summarise("phase-6, original phase6c model", R6c))
print("phase-6 original model: best %s p=%.2e" % (rows[-1]["best"], rows[-1]["best_p"]), flush=True)

# same original model on merged >= 0.9 — to separate panel from model
G9, ch9, pf9 = merged_panel(0.9)
R9c = dominance_scan(G9, ch9, pf9, samp, y_source="means", mother_fixed=False, use_Kd=False)
rows.append(summarise("merged >= 0.9, original phase6c model", R9c))
print("merged>=0.9 original model: best %s p=%.2e" % (rows[-1]["best"], rows[-1]["best_p"]), flush=True)

pd.DataFrame(rows).to_csv(SCR / "chr17_dom_scans.csv", index=False, encoding="utf-8-sig")
print("\nsaved:", SCR / "chr17_dom_scans.csv")
