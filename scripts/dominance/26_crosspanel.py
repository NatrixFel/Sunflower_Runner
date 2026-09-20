"""
CROSS-PANEL CHECK (second-group remark, item 4):
“verified” must not mean “verified in the set where it was found”.

Run the IDENTICAL dominance scan (mother fixed effects + K_a + K_d,
empirical threshold by parametric bootstrap) on a SECOND clean panel —
4_vcf/23_no_imputation.vcf (42 296 SNP, 0 % missing, 55 samples) —
and inspect the fate of both loci: chr14:169.21 and chr17:50.2–50.5.

For symmetry, the chr17 result on the merged panel is also printed.
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
import numpy as np, pandas as pd, allel
from scipy import stats, optimize
from pathlib import Path
import time, warnings
warnings.filterwarnings("ignore")

ROOT = DEPOSIT
HERE = OUT
CONF = {"LI29", "LI30"}
LTV = {"VK101A": "VK101", "VA761A": "VA761", "VK934A": "VK934"}
TRAIT = "seed_weight_1000"
NREP = 400

SEQ = {f"{i}.0": f"LI{i}" for i in range(1, 33)}
SEQ.update({"33.0": "LI34", "34.0": "LI35"})
SEQ.update({f"{i}.0": f"LI{i+2}" for i in range(35, 53)})
SEQ.update({"53.0": "LI33", "54.0": "LI36", "55.0": "VA761", "56.0": "VK101", "57.0": "VK934"})
def seqid(s):
    try: return SEQ.get(f"{int(float(s))}.0", str(s))
    except Exception: return str(s)

# ---------------- phase-6 panel ----------------
cs = allel.read_vcf(str(DEPOSIT/"23_no_imputation.vcf"),
                    fields=["samples","variants/CHROM","variants/POS","calldata/GT"])
na = allel.GenotypeArray(cs["calldata/GT"]).to_n_alt(fill=-1).astype(float)
na[na < 0] = np.nan
chrn = np.array([int(str(c).replace("CM00","").replace(".2",""))-7889 for c in cs["variants/CHROM"]])
pos = cs["variants/POS"]
names = np.array([seqid(s) for s in cs["samples"]])
print(f"phase-6 panel: {na.shape[0]:,} SNP x {na.shape[1]} samples; "
      f"missing {np.isnan(na).mean()*100:.4f} %")

af = np.nanmean(na,1)/2; maf = np.minimum(af,1-af)
keep = (chrn>=1)&(chrn<=17)&(maf>=0.05)
G = na[keep]; G = np.where(np.isnan(G), np.nanmean(G,1,keepdims=True), G)
chf = chrn[keep]; pf = pos[keep]; n_snp = G.shape[0]
s2c = {s:i for i,s in enumerate(names)}
print(f"  after MAF>=0.05: {n_snp:,} SNP")

# ---------------- phenotypes ----------------
bl = en_table(pd.read_csv(OUT / "22_hybrid_blup_phenotypes.csv"))
bl[["mom_long","father"]] = bl["hybrid"].str.split("_", expand=True)
bl["mom"] = bl["mom_long"].map(LTV)
d = bl.dropna(subset=[TRAIT]).copy()
d = d[~d["father"].isin(CONF)]
d = d[d["mom"].isin(s2c) & d["father"].isin(s2c)].reset_index(drop=True)
y = d[TRAIT].values.astype(float); n = len(d)
print(f"  hybrids with genotypes on this panel: n={n}  "
      f"(fathers {d['father'].nunique()}, mothers {d['mom'].nunique()})")

HG = np.empty((n, n_snp))
for i,row in d.iterrows():
    HG[i] = (G[:, s2c[row["mom"]]] + G[:, s2c[row["father"]]]) / 2
IH = (np.abs(HG-1) < 0.25).astype(float)
p_al = HG.mean(0)/2
Wa = HG - 2*p_al; Ka = (Wa@Wa.T)/(2*np.sum(p_al*(1-p_al)))
eh = 2*p_al*(1-p_al)
Wd = IH - eh; Kd = (Wd@Wd.T)/np.sum(eh*(1-eh))
X = np.column_stack([np.ones(n), (d["mom"]=="VA761").astype(float), (d["mom"]=="VK934").astype(float)])
useful = (IH.sum(0)>=5)&(IH.std(0)>1e-6)&(HG.std(0)>1e-6)
nX = X.shape[1]; df = n - nX - 2; I_n = np.eye(n)
print(f"  testable SNP: {int(useful.sum()):,}")

def reml_prof(h, yv):
    ha, hd = h
    if ha<0 or hd<0 or ha+hd>0.999: return 1e10
    A = ha*Ka + hd*Kd + (1-ha-hd)*I_n
    try: L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError: return 1e10
    AiX = np.linalg.solve(A, X); XtAX = X.T@AiX
    try: b = np.linalg.solve(XtAX, AiX.T@yv)
    except np.linalg.LinAlgError: return 1e10
    r = yv - X@b; q = r@np.linalg.solve(A, r)
    if q <= 0: return 1e10
    _, ldX = np.linalg.slogdet(XtAX)
    return 0.5*(2*np.sum(np.log(np.diag(L))) + ldX + (n-nX)*np.log(q/(n-nX)))

def fit_vc(yv, starts=((.5,.2),(.2,.5),(.05,.05))):
    best = None
    for st in starts:
        r = optimize.minimize(reml_prof, np.array(st), args=(yv,), method="Nelder-Mead",
                              options={"maxiter":400,"fatol":1e-7,"xatol":1e-5})
        if best is None or r.fun < best.fun: best = r
    ha, hd = np.clip(best.x, 0, None)
    if ha+hd > 0.999: s=(ha+hd)/0.999; ha,hd = ha/s, hd/s
    return ha, hd

def scan(yv, ha, hd):
    A = ha*Ka + hd*Kd + (1-ha-hd)*I_n
    ev, U = np.linalg.eigh(A); ev = np.maximum(ev,1e-10); sw = 1/np.sqrt(ev)
    Xw = (U.T@X)*sw[:,None]; Q,_ = np.linalg.qr(Xw)
    yw = (U.T@yv)*sw; yt = yw - Q@(Q.T@yw)
    Gt = (U.T@HG)*sw[:,None]; Gt -= Q@(Q.T@Gt)
    Ht = (U.T@IH)*sw[:,None]; Ht -= Q@(Q.T@Ht)
    gg = np.einsum('ij,ij->j',Gt,Gt); gg = np.where(gg<1e-12, np.nan, gg)
    gh = np.einsum('ij,ij->j',Gt,Ht); gy = Gt.T@yt
    Hp = Ht - Gt*(gh/gg); del Gt, Ht
    hh = np.einsum('ij,ij->j',Hp,Hp); hy = Hp.T@yt
    ok = useful & np.isfinite(hh) & (hh>1e-10) & np.isfinite(gg)
    b2 = hy/np.where(ok, hh, np.nan)
    sse = (yt@yt) - (gy**2)/gg - b2**2*hh
    sse = np.where(sse>1e-12, sse, np.nan)
    t = b2/np.sqrt(sse/df/hh)
    p = np.full(n_snp, np.nan); m = ok & np.isfinite(t)
    p[m] = 2*stats.t.sf(np.abs(t[m]), df)
    return p, b2

ha0, hd0 = fit_vc(y)
print(f"\nREML: Ka fraction={ha0:.3f}, Kd={hd0:.3f}, residual={1-ha0-hd0:.3f}")
obs, beta = scan(y, ha0, hd0)
vi = np.isfinite(obs)
lam = np.median(stats.chi2.isf(obs[vi],1))/stats.chi2.ppf(0.5,1)
bi = int(np.nanargmin(np.where(vi, obs, np.inf)))
print(f"lambda={lam:.2f}; tested {int(vi.sum()):,}")
print(f"BEST on the phase-6 panel: chr{int(chf[bi])}:{int(pf[bi])} p={obs[bi]:.3e} beta={beta[bi]:+.3f}")
print("top-10:")
for j in np.argsort(np.where(vi, obs, np.inf))[:10]:
    print(f"   chr{int(chf[j]):>2}:{int(pf[j]):>10}  p={obs[j]:.2e}  beta={beta[j]:+.2f}")

print("\nTARGET WINDOWS on the phase-6 panel:")
for nm,c,lo,hi in (("chr14:169.20-169.23",14,169_200_000,169_230_000),
                   ("chr17:50.20-50.50",17,50_200_000,50_500_000)):
    w = np.where((chf==c)&(pf>=lo)&(pf<=hi)&vi)[0]
    if not len(w): print(f"  {nm}: no testable SNP"); continue
    b = w[np.argmin(obs[w])]
    print(f"  {nm}: SNP={len(w):3d}  best chr{c}:{int(pf[b])} p={obs[b]:.3e} beta={beta[b]:+.2f}")

# ---------------- bootstrap ----------------
A0 = ha0*Ka + hd0*Kd + (1-ha0-hd0)*I_n
Lc = np.linalg.cholesky(A0 + 1e-10*I_n)
AiX = np.linalg.solve(A0, X); b0 = np.linalg.solve(X.T@AiX, AiX.T@y)
r0 = y - X@b0; s2 = (r0@np.linalg.solve(A0, r0))/(n-nX)
rng = np.random.default_rng(20260816)
minp = np.empty(NREP); t0=time.time()
for k in range(NREP):
    yv = X@b0 + np.sqrt(s2)*(Lc@rng.standard_normal(n))
    ha,hd = fit_vc(yv, starts=((ha0,hd0),(.05,.05)))
    pp,_ = scan(yv, ha, hd)
    minp[k] = np.nanmin(pp)
    if (k+1)%100==0:
        print(f"  bootstrap {k+1}/{NREP} ({time.time()-t0:.0f} s) thr5%={np.quantile(minp[:k+1],0.05):.2e}", flush=True)

thr5 = np.quantile(minp,0.05); thr1 = np.quantile(minp,0.01)
print(f"\n{'='*72}\nEMPIRICAL THRESHOLD on the phase-6 panel ({NREP} replicates): 5%={thr5:.3e}  1%={thr1:.3e}")
for nm,c,lo,hi in (("chr14:169.21",14,169_200_000,169_230_000),
                   ("chr17:50.2-50.5",17,50_200_000,50_500_000)):
    w = np.where((chf==c)&(pf>=lo)&(pf<=hi)&vi)[0]
    if not len(w): continue
    b = w[np.argmin(obs[w])]
    pe = (np.sum(minp<=obs[b])+1)/(NREP+1)
    print(f"  {nm:18s} p_dom={obs[b]:.3e}  empirical genome-wide p={pe:.4f}  -> "
          f"{'SIGNIFICANT' if pe<0.05 else 'NOT SIGNIFICANT'}")
pe_best = (np.sum(minp<=obs[bi])+1)/(NREP+1)
print(f"  best locus on the panel chr{int(chf[bi])}:{int(pf[bi])}  emp. p={pe_best:.4f}")

pd.DataFrame({"chr":chf,"pos":pf,"p_dom":obs,"beta_dom":beta}).to_csv(HERE/"crosspanel_phase6_scan.csv", index=False)
pd.DataFrame({"min_p":minp}).to_csv(HERE/"crosspanel_phase6_null.csv", index=False)
print("\nSaved: crosspanel_phase6_scan.csv, crosspanel_phase6_null.csv")
