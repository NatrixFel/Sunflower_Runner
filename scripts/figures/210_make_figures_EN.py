# -*- coding: utf-8 -*-
"""
English versions of Figs. 1–7 (Frontiers, PNG 600 dpi).

Ported from the 2026-08-17 figure pack (md5 c251f604…): only
input paths and caption strings were changed. Data, filters, trait order and panel
geometry were left untouched, so the English figures are numerically identical to the previous pack.

Numbering follows first mention in the English manuscript (renumbering 2026-08-19):
Fig1 = pack 2, Fig2 = pack 3, Fig3 = pack 6, Fig4 = pack 1, Fig5 = pack 7,
Fig6 = pack 4, Fig7 = pack 5.

Two substantive differences from the pack. Power figure (Fig5): the “hybrids” vertical is drawn
at n = 162, not 159 — the canonical denominator, DECISIONS D005, which superseded D002.
Panel 1B: plot-level variance-component shares (D008) instead of the withdrawn
Baker ratio 0.71 / 0.94 / 0.94 of the old pooled estimate.

Okabe-Ito palette (colorblind-safe). Output: manuscriptigures\Fig1..Fig7_EN.png
"""
from __future__ import annotations

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
import numpy as np, pandas as pd
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = DEPOSIT
DATA = DEPOSIT
FIGDIR = OUT / "figures"
FIG = FIGDIR; FIG.mkdir(parents=True, exist_ok=True)

# --- Okabe-Ito ---
OI = {"black":"#000000","orange":"#E69F00","sky":"#56B4E9","green":"#009E73",
      "yellow":"#F0E442","blue":"#0072B2","verm":"#D55E00","purple":"#CC79A7"}
OIL, CONF = OI["blue"], OI["verm"]
MOTH_COL = {"VK101A":OI["blue"],"VA761A":OI["orange"],"VK934A":OI["green"]}
MOTH_EN  = {"VK101A":"VK101A","VA761A":"VA761A","VK934A":"VK934A"}
def en_line(name: str) -> str:
    return en_id(name)

# --- Frontiers profile ---------------------------------------------------------
# Single-column width 85 mm, two-column 180 mm; DPI is at the FINAL size, so
# the figure size is set immediately in finished millimetres, and type sizes and line
# widths are in points that will result at that size. The grid is removed: a grid line
# thinner than 2 pt would violate the requirement, and a 2 pt grid would swamp the data.
MM = 1 / 25.4
W1, W2 = 85 * MM, 180 * MM        # one and two columns, in inches
LW = 2.0                          # minimum line width, pt
FS_MIN = 8                        # minimum type size, pt
# Resolution is a single number and is applied to all three formats and to PIL. Frontiers requires
# at least 300 dpi at final size; from 2026-09-20 (author decision, wave 15) the whole set
# is built at 600 — the extra costs nothing, and a mixed set at submission is undesirable.
# Type sizes and widths are in points and do not depend on resolution: at 600 dpi they are still 8 pt and 2 pt.
DPI = 600
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
    "axes.spines.right":False,"axes.grid":False,
    "axes.linewidth":LW,"xtick.major.width":LW,"ytick.major.width":LW,
    "xtick.minor.width":LW,"ytick.minor.width":LW,"lines.linewidth":LW,
    "patch.linewidth":LW,"legend.fontsize":FS_MIN,"axes.labelsize":9,"axes.titlesize":9,
    "xtick.labelsize":FS_MIN,"ytick.labelsize":FS_MIN,
    "axes.axisbelow":True,"figure.dpi":DPI,"savefig.dpi":DPI,"savefig.bbox":None})

FIGNUM = {"Fig1_combining_ability_EN.png":1, "Fig2_genomic_prediction_EN.png":2,
          "Fig3_heterosis_distance_EN.png":3, "Fig4_structure_oil_EN.png":4,
          "Fig5_power_EN.png":5, "Fig6_use_type_artifact_EN.png":6,
          "Fig7_separate_implementation_chr8_EN.png":7}

def save(fig, name, layout=True):
    """PNG — for the docx build; TIFF (LZW) and JPEG — for submission.

    bbox="tight" is NOT used: it crops the canvas, and the width is then no longer exactly
    85 or 180 mm, while resolution at final size falls below the declared value. Instead
    tight_layout is used inside a fixed canvas. The alpha channel is stripped: Frontiers
    requires RGB.
    """
    if layout:
        try:
            fig.tight_layout()
        except Exception:
            pass
    num = FIGNUM[name]
    fig.savefig(FIG / name, bbox_inches=None)
    tif, jpg = FIG / ("Figure%d.tiff" % num), FIG / ("Figure%d.jpg" % num)
    fig.savefig(tif, bbox_inches=None, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(jpg, bbox_inches=None, pil_kwargs={"quality": 95, "subsampling": 0})
    from PIL import Image
    with Image.open(tif) as im:
        if im.mode != "RGB":
            rgb = im.convert("RGB")
            rgb.save(tif, compression="tiff_lzw", dpi=(DPI, DPI))
    plt.close(fig)
    print("  ->", name, "+ Figure%d.tiff/.jpg" % num)

def cn(c):
    try: return int(c.replace("CM00","").replace(".2",""))-7889
    except: return -1

# ================= data =================
d = np.load(DATA/"23_merged_genotypes.npz", allow_pickle=True)
G = d["genotypes"].astype(float); G[G<0]=np.nan
chrom = np.array([cn(c) for c in d["chrom"]]); samp=en_ids(d["samples"].astype(str))
cr=np.mean(~np.isnan(G),1); af=np.nanmean(G,1)/2; maf=np.minimum(af,1-af)
keep=(chrom>=1)&(chrom<=17)&(cr>=0.5)&(maf>=0.05)   # ordination panel: 161 791 SNP
Gk=G[keep]; Gk=np.where(np.isnan(Gk),np.nanmean(Gk,1,keepdims=True),Gk)
print("ordination panel (Fig4) and distances (Fig3):", Gk.shape[0], "SNP")
fathers=[s for s in samp if is_line_id(s)]; MO={"VK101A":"VK101","VA761A":"VA761","VK934A":"VK934"}

lines_v2=en_table(pd.read_parquet(OUT / "22_lines_tidy.parquet"))
oilph=lines_v2[lines_v2["trait"]=="oil_content"].groupby("genotype")["value"].mean()
hm=en_table(pd.read_csv(INTERMEDIATE/"22_hybrid_means_delivered.csv"))
# means of 162 hybrids from plot-level raw data: used for
# variance components (Fig1B) and the distance–heterosis association (Fig3)
_plot=en_table(pd.read_parquet(OUT / "22_hybrid_plot_level.parquet"))
hm162=(_plot.groupby(["mother","father","trait"])["value"].mean()
            .unstack("trait").reset_index())
print("hybrid means (plot-level):", len(hm162))

# ============ Fig4 (pack 1). PCA structure of lines + oil content ============
fi=[samp.index(f) for f in fathers]; Xf=Gk[:,fi].T
Xc=(Xf-Xf.mean(0)); Xc/=(Xc.std(0)+1e-9)
U,S,_=np.linalg.svd(Xc,full_matrices=False); pcs=U[:,:2]*S[:2]
pev=(S**2/np.sum(S**2))[:2]*100
isconf=np.array([f in ("LI29","LI30") for f in fathers])
# SVD leaves the sign of each component free. Flip so the confectionery pair
# sits in the upper-left, as in the submitted panel, and so LI29 (higher PC2)
# is not given the offset written for the lower point.
if pcs[fathers.index("LI29"), 1] < 0:
    pcs[:, 1] *= -1
if pcs[fathers.index("LI29"), 0] > 0:
    pcs[:, 0] *= -1
fig,axs=plt.subplots(1,2,figsize=(W2,W2*0.48),
                     gridspec_kw={"width_ratios":[1.0,1.28],"wspace":0.28})
ax=axs[0]
ax.scatter(pcs[~isconf,0],pcs[~isconf,1],c=OIL,s=28,label="oilseed (52)",linewidth=0)
ax.scatter(pcs[isconf,0],pcs[isconf,1],c=CONF,s=38,marker="D",label="confectionery (2)",edgecolor="black",linewidth=LW,zorder=5)
# LI29 is the upper confectionery point, LI30 the lower.
conf_off={"LI29":(8,6),"LI30":(8,-10)}
conf_va={"LI29":"bottom","LI30":"top"}
for i,f in enumerate(fathers):
    if isconf[i]:
        ax.annotate(en_line(f),(pcs[i,0],pcs[i,1]),fontsize=8,
                    xytext=conf_off.get(f,(8,4)),textcoords="offset points",
                    ha="left",va=conf_va.get(f,"center"))
ax.set_xlabel(f"PC1 ({pev[0]:.1f}%)"); ax.set_ylabel(f"PC2 ({pev[1]:.1f}%)")
ax.set_title("(A) Genomic structure of the 54 paternal lines")
ax.legend(frameon=False,fontsize=FS_MIN,loc="lower left")
ax=axs[1]
oil_vals=oilph.reindex(fathers).values
ax.hist(oil_vals[~isconf],bins=14,color=OIL,alpha=0.85,label="oilseed")
ax.hist(oil_vals[isconf],bins=6,color=CONF,alpha=0.9,label="confectionery")
ax.axvspan(28,30,color=CONF,alpha=0.12)
ax.set_xlim(27, 54)
ax.set_xlabel("Line oil content (%)"); ax.set_ylabel("Number of lines")
ax.set_title("(B) Bimodality of oil content")
# Legend sits in the empty 31–39% valley, top of the panel: higher than the
# bars, no clipped labels, and the x-axis can use the full panel width.
ax.legend(frameon=False,fontsize=FS_MIN,loc="upper left",bbox_to_anchor=(0.20,0.98),
          borderaxespad=0.0,handlelength=1.15)
fig.subplots_adjust(left=0.075,right=0.99,top=0.90,bottom=0.15,wspace=0.24)
save(fig,"Fig4_structure_oil_EN.png",layout=False)

# ============ Fig1 (pack 2). Combining ability ============
gca=en_table(pd.read_csv(OUT / "25_gca_fathers.csv")).set_index("father")

# Panel B: plot-level variance-component shares, tester fixed.
# Five components: paternal GCA, SCA, Year x Line, Year x Tester x Line, experimental error.
# Values are NOT recalculated here — they are read from brain\25_baker_v4_intervals.json,
# written by scripts\25_baker_v4_intervals.py (REML with full-rank X, bounds
# gamma >= 0, profile interval of the ratio, share SEs by the delta method).
# So the figure and the text cannot diverge.
import json as _json
_iv=_json.loads((ROOT/"brain"/"25_baker_v4_intervals.json").read_text(encoding="utf-8"))
BT=[("seed_yield","seed yield"),("oil_content","oil content"),("seed_weight_1000","1000-seed weight")]
COMP=[(0,"paternal GCA",OI["blue"]),(1,"SCA",OI["orange"]),
      (2,"year x line",OI["sky"]),(3,"year x tester x line",OI["purple"]),
      (4,"experimental error","#7F7F7F")]
vc={}
for tr,_en in BT:
    z=_iv[tr]
    vc[tr]=dict(share={i:z["shares"][i] for i,_,_ in COMP},
                err  ={i:1.96*z["share_se"][i] for i,_,_ in COMP},
                baker=z["baker"], lo=z["ci_low"], hi=z["ci_high"])
    print(f"  Fig1B {tr}: shares " + ", ".join(f"{lab} {100*vc[tr]['share'][i]:.1f}%" for i,lab,_ in COMP)
          + f"; Baker {z['baker']:.3f} ({z['ci_low']:.2f}-{z['ci_high']:.2f})")
fig,axs=plt.subplots(1,2,figsize=(W2,W2*0.52))
ax=axs[0]
gu=gca["seed_yield"].sort_values()
cols=[CONF if f in ("LI29","LI30") else OIL for f in gu.index]
ax.bar(range(len(gu)),gu.values,color=cols,width=0.9)
ax.axhline(0,color="black",linewidth=LW)
top=gu.sort_values(ascending=False).head(3)
# Wave 14 (R4): previously the three rotated labels stood in a staircase (+3, +29, +55 pt) and the top one
# spilled past the axis onto the panel title. They now sit at one height (+3 pt above the bar),
# spread horizontally by ±5 pt (three adjacent bars ~4 pt wide, label ~9 pt), and
# the top of the axis is raised so the labels lie entirely inside the panel. Data untouched.
for k,f in enumerate(top.index):
    x=list(gu.index).index(f)
    ax.annotate(en_line(f),(x,gu[f]),fontsize=FS_MIN,ha="center",va="bottom",
                rotation=90,xytext=(5-5*k,3),textcoords="offset points")
ax.set_ylim(gu.min()-0.06, gu.max()+0.34)
ax.set_xlabel("Paternal lines (ranked by GCA)"); ax.set_ylabel("GCA for seed yield (t/ha)")
ax.set_title("(A) Paternal GCA for seed yield"); ax.set_xticks([])
ax=axs[1]
xg=np.arange(len(BT)); w=0.16
for j,(kk,lab,col) in enumerate(COMP):
    vals=[100*vc[tr]["share"][kk] for tr,_ in BT]
    errs=[100*vc[tr]["err"][kk]   for tr,_ in BT]
    pos=xg+(j-2.0)*w
    ax.bar(pos,vals,w,color=col,label=lab)
    ax.errorbar(pos,vals,yerr=errs,fmt="none",ecolor="black",elinewidth=LW,capsize=2.0)
    for x,v,e in zip(pos,vals,errs):
        ax.annotate(f"{v:.0f}",(x,v+e),ha="center",va="bottom",fontsize=FS_MIN,xytext=(0,1.5),
                    textcoords="offset points")
ax.axhline(0,color="black",linewidth=LW)
_uy,_ol,_ms=vc["seed_yield"],vc["oil_content"],vc["seed_weight_1000"]
BAK_LAB=[f"seed yield\nBaker {_uy['baker']:.3f}\n({_uy['lo']:.2f}-{_uy['hi']:.2f})",
         f"oil content\nBaker {_ol['baker']:.3f}",
         f"1000-seed\nweight\nBaker {_ms['baker']:.3f}"]   # W14: two lines — otherwise the label overlaps "oil content" (measured: 11 px)
ax.set_xticks(xg); ax.set_xticklabels(BAK_LAB,fontsize=FS_MIN)
ax.set_ylim(-16,132); ax.set_yticks([0,20,40,60,80,100])
ax.set_xlabel("Trait (Baker's ratio printed under each)")
ax.set_ylabel("Share of the five components (%)")
ax.set_title("(B) Variance components by trait")
ax.legend(frameon=False,fontsize=FS_MIN,ncol=2,loc="upper left",bbox_to_anchor=(0.02,1.005),
          columnspacing=0.6,handlelength=1.0,handletextpad=0.3)
save(fig,"Fig1_combining_ability_EN.png")

# ============ Fig2 (pack 3). Genomic prediction ============
gs=en_table(pd.read_csv(OUT / "27_gs_accuracy_blup.csv",comment="#"))
order=["seed_yield","oil_yield","oil_content","seed_weight_1000"]
gs=gs.set_index("trait").reindex(order).reset_index()
# The sqrt(H2) ceiling is NOT taken from 27_gs_accuracy_blup.csv: those H2 values were computed
# under a model without the “genotype x year” term, withdrawn by decision D020. Recalculated
# values are read from brain\22_h2_recalc.json (scripts\22_h2_recalc.py) so the figure and text cannot diverge.
_h2=_json.loads((ROOT/"brain"/"22_h2_recalc.json").read_text(encoding="utf-8"))
gs["ceiling_sqrt_H2"]=[np.sqrt(_h2[t]["new"]["H2_mean"]) if t in _h2 else np.nan
                        for t in gs["trait"]]
print("  Fig2 sqrt(H2) ceilings, D020 recalc: " + ", ".join(
    f"{t} {v:.3f}" for t,v in zip(gs['trait'],gs['ceiling_sqrt_H2']) if not np.isnan(v)))
fig,ax=plt.subplots(figsize=(W1,W1*0.92))
x=np.arange(len(gs)); w=0.55
ax.bar(x,gs["GS_accuracy_CV1"],w,color=OI["blue"],label="predictive ability (CV1)")
ceil=gs["ceiling_sqrt_H2"].values
for i,c in enumerate(ceil):
    if not np.isnan(c): ax.plot([i-w/2,i+w/2],[c,c],color=OI["verm"],linewidth=LW+0.5)
for i,v in enumerate(gs["GS_accuracy_CV1"]): ax.annotate(f"{v:.2f}",(i,v),ha="center",va="bottom",fontsize=FS_MIN)
ax.plot([],[],color=OI["verm"],linewidth=LW+0.5,label="ceiling $\\sqrt{H^2}$")
ax.set_xticks(x); ax.set_xticklabels(["seed\nyield","oil\nyield","oil\ncontent","1000-seed\nweight"],fontsize=FS_MIN)
ax.set_xlabel("Trait")
ax.set_ylabel("Predictive ability, r (predicted vs observed)")
# legend ceiling raised above the data: at ylim 1.05 the legend covered the yield-ceiling mark
ax.set_ylim(0,1.34); ax.set_yticks(np.arange(0,1.01,0.2))
ax.set_title("Predictive ability by trait\n(hybrid level, leave-one-father-out CV1)",fontsize=9); ax.legend(frameon=False,fontsize=FS_MIN,loc="upper left")
save(fig,"Fig2_genomic_prediction_EN.png")

# ============ Fig6 (pack 4). Use-type artefact ============
ORDER = [("oil_content","oil content"),("hull_content","hull content"),("seed_weight_1000","1000-seed weight"),
         ("head_diameter","head diameter"),("plant_height","plant height"),
         ("days_emergence_flowering","emergence-to-flowering"),
         ("autofertility_self","seed set (selfed)"),("autofertility_open","seed set (open)")]
N_AFFECTED = 3
FLOOR = 0.35

def counts(csv, col="n_below_emp"):
    am = en_table(pd.read_csv(OUT/csv))
    def c(tr, nab):
        r = am[(am["trait"]==tr) & (am["callset"]==nab)]
        return int(r[col].values[0]) if len(r) else 0
    return ([c(t,"54_with_confectionery") for t,_ in ORDER],
            [c(t,"52_oilseed_only") for t,_ in ORDER])

PANELS = [("26_artifact_merged.csv",    "(A) EMMAX without principal-component covariates\n(relatedness still controlled by K$_a$ alone)"),
          ("26_artifact_merged_pc.csv", "(B) EMMAX, declared model\n(three principal components + K$_a$)")]

fig, axs = plt.subplots(1, 3, figsize=(W2,W2*0.95), sharey=True,
                        gridspec_kw={"width_ratios":[1,1,0.5]})
for ax,(csv,title) in zip(axs[:2], PANELS):
    v54, v52 = counts(csv)
    x = np.arange(len(ORDER)); w = 0.38
    ax.bar(x-w/2, np.maximum(v54,FLOOR), w, color=CONF, label="54 lines (with confectionery)")
    ax.bar(x+w/2, np.maximum(v52,FLOOR), w, color=OIL,  label="52 lines (without confectionery)")
    for i,(a,b) in enumerate(zip(v54,v52)):
        ax.annotate(str(a),(i-w/2,max(a,FLOOR)),ha="center",va="bottom",fontsize=8,
                    fontweight="bold" if a>0 else "normal")
        ax.annotate(str(b),(i+w/2,max(b,FLOOR)),ha="center",va="bottom",fontsize=8,
                    fontweight="bold" if b>0 else "normal")
    ax.axvline(N_AFFECTED-0.5, color=OI["black"], linewidth=LW, linestyle=":")
    ax.set_yscale("log"); ax.set_ylim(FLOOR, 600)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n"," ") for _,n in ORDER], fontsize=FS_MIN,
                       rotation=90, ha="center", va="top")
    ax.set_xlim(-0.7, len(ORDER)-0.3)
    ax.set_xlabel("Trait", fontsize=9)
    ax.set_title(f"{title}\ntotal at 54 lines: {sum(v54)} -> at 52: {sum(v52)}", fontsize=FS_MIN)
axs[0].set_ylabel("Number of SNPs below the empirical\ngenome-wide threshold (log scale)")
axs[0].annotate("tied to\nuse type", ((N_AFFECTED-1)/2, 330), ha="center", va="center",
                fontsize=FS_MIN, color=CONF, fontweight="bold")   # W14: two lines — a single line spilled onto the axis and the dashed line
axs[1].annotate("negative control", ((N_AFFECTED-0.5+len(ORDER)-0.5)/2, 330), ha="center",
                fontsize=FS_MIN, color=OI["black"])
h, lab = axs[0].get_legend_handles_labels()
fig.legend(h, lab, frameon=False, fontsize=FS_MIN, ncol=2, loc="lower center",
           bbox_to_anchor=(0.5, 0.005))

bk = en_table(pd.read_csv(OUT/"26_canonical_blink.csv"))
bk = bk[bk["model"].str.contains("canon")]
def bsum(nab): return int(bk[bk["panel"] == nab]["BLINK_n_p<5e-8"].sum())
axc = axs[2]
xg = np.arange(2); w2 = 0.38
v54c = [0, bsum("54_with_confectionery")]
v52c = [0, bsum("52_oilseed_only")]
axc.bar(xg-w2/2, np.maximum(v54c,FLOOR), w2, color=CONF)
axc.bar(xg+w2/2, np.maximum(v52c,FLOOR), w2, color=OIL)
for i,(a,b) in enumerate(zip(v54c,v52c)):
    axc.annotate(str(a),(i-w2/2,max(a,FLOOR)),ha="center",va="bottom",fontsize=8.5,
                 fontweight="bold" if a>0 else "normal")
    axc.annotate(str(b),(i+w2/2,max(b,FLOOR)),ha="center",va="bottom",fontsize=8.5,
                 fontweight="bold" if b>0 else "normal")
axc.set_yscale("log"); axc.set_ylim(FLOOR, 600)
axc.set_xticks(xg); axc.set_xticklabels(["EMMAX","BLINK"], fontsize=FS_MIN)
axc.set_xlabel("Association model", fontsize=9)
axc.set_xlim(-0.65, 1.65)
axc.set_title("(C) Declared model,\nthreshold 5 x 10$^{-8}$\n"
              f"EMMAX 0, BLINK {bsum('54_with_confectionery')}", fontsize=FS_MIN)

fig.suptitle("The use-type artefact and its standard control:\ntwo confectionery lines out of 54",
             y=0.995, fontsize=9)
fig.subplots_adjust(left=0.115,right=0.965,top=0.80,bottom=0.345,wspace=0.14)
save(fig,"Fig6_use_type_artifact_EN.png",layout=False)

# ============ Fig7 (pack 5). Sensitivity of Chr8:782,740 to sample composition ============
# Redone 2026-08-21 from a co-author prompt: points with connecting lines instead of bars,
# short legend labels, panel details moved into the figure caption, interpretive phrases
# (“ordinary marker”, “leading hit”, “125th genome-wide”) removed from the title.
# Data are NOT recalculated. Point heights are the registered p-values:
#   GAPIT-BLINK 1.034022e-12 and 3.463436e-4 (N-61, GAPIT outputs of both runs);
#   our EMMAX — from 26_chr8_canonical_check.csv (6.28e-4 at 54 lines, 2.495e-3 at 52).
# Panel sizes come from the same file: 81 903 at 54 lines, 80 410 at 52.
ch8 = en_table(pd.read_csv(OUT/"26_chr8_canonical_check.csv"))
def our(nab, col="EMMAX_p"):
    return float(ch8[ch8["panel"] == nab][col].values[0])
def npanel(nab):
    return int(ch8[ch8["panel"] == nab]["n_SNP"].values[0])
def fmt_p(pv):
    e = int(np.floor(np.log10(pv))); m = pv / 10.0**e
    return "$p$ = %.1f $\\times$ 10$^{%d}$" % (m, e)

GAP54, GAP52 = 1.034022e-12, 3.463436e-4
SERIES = [("GAPIT-BLINK", [GAP54, GAP52], CONF, "o", +1),
          ("EMMAX",
           [our("54_with_confectionery"), our("52_oilseed_only")], OIL, "s", -1)]
print("  Fig7: panels %d and %d SNP; p = %s" % (
    npanel("54_with_confectionery"), npanel("52_oilseed_only"),
    ", ".join("%.3g" % v for v in [GAP54, GAP52, our("54_with_confectionery"), our("52_oilseed_only")])))

fig, ax = plt.subplots(figsize=(W2, W2*0.52))
xpos = np.arange(2)
for lab, ps, col, mk, side in SERIES:
    y = [-np.log10(pv) for pv in ps]
    ax.plot(xpos, y, color=col, marker=mk, markersize=7, linewidth=2.2,
            label=lab, zorder=3, clip_on=False)
    for xi, yi, pv in zip(xpos, y, ps):
        ax.annotate(fmt_p(pv), (xi, yi), fontsize=FS_MIN, ha="center",
                    va="bottom" if side > 0 else "top",
                    xytext=(0, 7*side), textcoords="offset points")

THR = -np.log10(5e-8)                      # 7.301
ax.axhline(THR, color="#4D4D4D", linestyle="--", linewidth=LW, zorder=1)
ax.annotate("Nominal threshold: 5 $\\times$ 10$^{-8}$", (1.45, THR), fontsize=FS_MIN,
            ha="right", va="bottom", xytext=(0, 4), textcoords="offset points", color="#4D4D4D")

ax.set_xticks(xpos)
ax.set_xticklabels(["54 lines\n(with confectionery)", "52 lines\n(oilseed only)"], fontsize=FS_MIN)
ax.set_xlim(-0.45, 1.45)
ax.set_ylim(0, 13); ax.set_yticks(np.arange(0, 13.1, 2))
ax.set_xlabel("Sample composition")
ax.set_ylabel("$-$log$_{10}$($p$) for Chr8:782,740 (oil content)")
ax.set_title("Sensitivity of the Chr8:782,740 association to sample composition", fontsize=9)
ax.legend(frameon=False, fontsize=FS_MIN, loc="upper right")
save(fig,"Fig7_separate_implementation_chr8_EN.png")

# ============ Fig3 (pack 6). Distance-based heterosis by mother ============
s2c={s:i for i,s in enumerate(samp)}
def f1d(m,f):
    gm=np.round(Gk[:,s2c[MO[m]]]); gf=np.round(Gk[:,s2c[f]]); return np.mean(np.abs(gm-gf)/2)
hm2=hm162.copy(); hm2["dist"]=[f1d(m,f) for m,f in zip(hm2["mother"],hm2["father"])]
fig,axs=plt.subplots(1,3,figsize=(W2,W2*0.42),sharey=True)
for pl,(ax,mom) in zip("ABC", zip(axs,["VK101A","VA761A","VK934A"])):
    sub=hm2[hm2["mother"]==mom]; x=sub["dist"].values; y=sub["seed_yield"].values
    m=~np.isnan(y); r,p=stats.pearsonr(x[m],y[m])
    ax.scatter(x,y,c=MOTH_COL[mom],s=26,alpha=0.85,linewidth=0)
    b=np.polyfit(x[m],y[m],1); xs=np.array([x[m].min(),x[m].max()])
    ax.plot(xs,b[0]*xs+b[1],color="black",linewidth=LW,
            linestyle="-" if p<0.05 else "--")
    ax.set_title(f"({pl}) {MOTH_EN[mom]}\nr = {r:.2f}, p = {p:.4f}"+(" *" if p<0.05 else ""),fontsize=9)
axs[0].set_ylabel("Hybrid seed yield (t/ha)")
fig.supxlabel("Mother-father genetic distance (expected F1 heterozygosity, share of loci)",
              fontsize=9,y=0.02)
fig.suptitle("Distance-based heterosis is mother-specific (significant only for VK934A)",y=0.99,fontsize=9)
fig.subplots_adjust(left=0.10,right=0.99,top=0.80,bottom=0.22,wspace=0.10)
save(fig,"Fig3_heterosis_distance_EN.png",layout=False)

# ============ Fig5 (pack 7). Power analysis ============
fig,ax=plt.subplots(figsize=(W1,W1*0.92))
alpha=5e-8
from scipy.stats import ncx2,chi2
thr=chi2.isf(alpha,1)
ns=np.arange(30,801,5)
for pve,col in [(0.05,OI["sky"]),(0.10,OI["blue"]),(0.20,OI["green"]),(0.42,OI["orange"])]:
    R2=pve; ncp=ns*R2/(1-R2); power=ncx2.sf(thr,1,ncp)
    ax.plot(ns,power,color=col,linewidth=LW,label=f"PVE = {int(pve*100)}%")
ax.axhline(0.8,color="black",linestyle=":",linewidth=LW); ax.annotate("80% power",(770,0.82),fontsize=FS_MIN,ha="right")
ax.axvline(54,color=CONF,linestyle="--",linewidth=LW); ax.annotate("our lines (n = 54)",(62,0.05),fontsize=FS_MIN,rotation=90,va="bottom")
ax.axvline(162,color=OI["purple"],linestyle="--",linewidth=LW); ax.annotate("hybrids (n = 162)",(172,0.05),fontsize=FS_MIN,rotation=90,va="bottom")
ax.set_xlabel("Sample size (n)"); ax.set_ylabel(r"Power ($\alpha$ = 5 x 10$^{-8}$)"); ax.set_ylim(0,1)
ax.set_title("Power ceiling: at n $\\approx$ 50\nonly PVE > 42% is detectable",fontsize=9)
ax.legend(frameon=False,fontsize=FS_MIN,title="locus variance explained",title_fontsize=FS_MIN)
save(fig,"Fig5_power_EN.png")

print("\nAll figures in", FIG)
