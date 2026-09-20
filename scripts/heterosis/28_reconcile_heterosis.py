"""
Reconciliation of the heterosis metric: distance↔yield by mother and overall.
The same metric (as in Phase 9a: f1_het = mean(|round(gm)-round(gd)|/2)) on three genotypes:
var1 (reproduces the stored r=0.46), merged (phase11 canon), + var2 from the re-check.
The aim is to separate the VCF effect from the metric effect and lock one canonical result.
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
import warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import allel
warnings.filterwarnings("ignore")

ROOT = work()  # undeposited working tree
SEQ={};
for n in range(1,33): SEQ[n]=f"LI{n}"
SEQ[33]="LI34"; SEQ[34]="LI35"
for n in range(35,53): SEQ[n]=f"LI{n+2}"
SEQ[53]="LI33"; SEQ[54]="LI36"; SEQ[55]="VA761"; SEQ[56]="VK101"; SEQ[57]="VK934"
LONG={"VK101A":"VK101","VA761A":"VA761","VK934A":"VK934"}
def cn(c):
    try: return int(c.replace("CM00","").replace(".2",""))-7889
    except: return -1

def f1_dist(gm, gd):
    return np.mean(np.abs(np.round(gm)-np.round(gd))/2.0)

def filt_impute(gt, chr_num, maf_min=0.05, cr_min=0.5):
    cr=np.mean(~np.isnan(gt),axis=1); af=np.nanmean(gt,axis=1)/2; maf=np.minimum(af,1-af)
    keep=(chr_num>=1)&(chr_num<=17)&(cr>=cr_min)&(maf>=maf_min)
    g=gt[keep]; g=np.where(np.isnan(g),np.nanmean(g,axis=1,keepdims=True),g); return g

# ---- var1 (LI15/LI51 consensus) ----
def load_var1():
    vcf=ROOT/"6_новые данные_ответы + исходники +vcf"/"sunflower_var1_bam_merged_15_49.vcf"
    cb=allel.read_vcf(str(vcf),fields=["samples","variants/CHROM","variants/POS","variants/ALT","calldata/GT"])
    sr=en_ids(cb["samples"].astype(str)); nalt=allel.GenotypeArray(cb["calldata/GT"]).to_n_alt(fill=-1).astype(float); nalt[nalt<0]=np.nan
    chr_num=np.array([cn(c) for c in cb["variants/CHROM"]]); alt=cb["variants/ALT"]; bi=(alt[:,0]!="")&(alt[:,1]=="")
    ci={n:i for i,n in enumerate(sr)}
    r15=[i for i,s in enumerate(sr) if s.startswith("15-")]; r49=[i for i,s in enumerate(sr) if s.startswith("49-")]
    canon=[SEQ[s] for s in range(1,58)]; n57=np.full((nalt.shape[0],57),np.nan)
    for j,name in enumerate(canon):
        if name=="LI15": n57[:,j]=np.nanmean(nalt[:,r15],axis=1)
        elif name=="LI51": n57[:,j]=np.nanmean(nalt[:,r49],axis=1)
        else:
            seq=[k for k,v in SEQ.items() if v==name][0]; n57[:,j]=nalt[:,ci[f"{seq}.0"]]
    g=filt_impute(n57[bi], chr_num[bi]); return g, canon

# ---- var2 ----
def load_var2():
    vcf=ROOT/"6_новые данные_ответы + исходники +vcf"/"sunflower_var2_vcf_merged.vcf"
    cb=allel.read_vcf(str(vcf),fields=["samples","variants/CHROM","calldata/GT"])
    nalt=allel.GenotypeArray(cb["calldata/GT"]).to_n_alt(fill=-1).astype(float); nalt[nalt<0]=np.nan
    chr_num=np.array([cn(c) for c in cb["variants/CHROM"]])
    names=[SEQ[int(float(s))] for s in cb["samples"]]
    order=[names.index(SEQ[s]) for s in range(1,58)]
    g=filt_impute(nalt[:,order], chr_num); return g, [SEQ[s] for s in range(1,58)]

# ---- merged (phase11 canon) ----
def load_merged():
    d=np.load(DEPOSIT / "23_merged_genotypes.npz", allow_pickle=True)
    g=d["genotypes"].astype(float); g[g<0]=np.nan
    chr_num=np.array([cn(c) for c in d["chrom"]]); samp=en_ids(d["samples"].astype(str))
    g=filt_impute(g, chr_num); return g, samp

# ---- phenotypes (as in Phase 9a) ----
hyb=en_table(pd.read_parquet(ROOT/"phase1_output"/"01_hybrids_tidy.parquet"))
piv=(hyb[hyb["trait"].isin(["seed_yield","oil_content"])]
     .groupby(["mother_long","father","trait"])["value"].mean().unstack().reset_index())
piv["oil_yield"]=piv["seed_yield"]*piv["oil_content"]/100.0

def analyze(g, samp, tag):
    s2c={s:i for i,s in enumerate(samp)}
    sub=piv[piv["mother_long"].map(LONG).isin(samp) & piv["father"].isin(samp)].copy()
    sub["dist"]=[f1_dist(g[:,s2c[LONG[m]]], g[:,s2c[f]]) for m,f in zip(sub["mother_long"],sub["father"])]
    rows=[]
    for scope in ["ALL"]+list(LONG.keys()):
        d=sub if scope=="ALL" else sub[sub["mother_long"]==scope]
        m=d["seed_yield"].notna()
        ru,pu=stats.pearsonr(d.loc[m,"dist"],d.loc[m,"seed_yield"])
        ro,po=stats.pearsonr(d.loc[m,"dist"],d.loc[m,"oil_yield"])
        rows.append({"VCF":tag,"group":scope,"n":int(m.sum()),
                     "r_dist_yield":round(ru,3),"p_yield":round(pu,3),
                     "r_dist_oil":round(ro,3),"p_oil":round(po,3)})
    return rows

allrows=[]
print("=== merged (phase11 canon) ===", flush=True)
gm_,sm_=load_merged(); print(f"  {gm_.shape[0]:,} SNP × {gm_.shape[1]}"); allrows+=analyze(gm_,sm_,"merged")
print("=== var1 (bam-merge, reproduces the stored result) ===", flush=True)
g1,s1=load_var1(); print(f"  {g1.shape[0]:,} SNP"); allrows+=analyze(g1,s1,"var1")
print("=== var2 (session) ===", flush=True)
g2,s2=load_var2(); print(f"  {g2.shape[0]:,} SNP"); allrows+=analyze(g2,s2,"var2")

res=pd.DataFrame(allrows)
res.to_csv(OUT/"28_reconcile_heterosis.csv", index=False, encoding="utf-8-sig")
print("\n=== RECONCILIATION distance↔yield (one metric, 3 genotypes) ===")
print(res.to_string(index=False))
