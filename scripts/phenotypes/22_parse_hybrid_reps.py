"""
Phase 14a — parse hybrid raw data with replicates + outlier QC + h²/repeatability.

Input: 1_данные из вниимка/гибриды по повт-тям 2022-2023 итог гг..xlsx
  Sheets '2022','2023': yield/oil/seed weight × 3 replicates (plot level).
  Sheets 'Биометрия 2022/2023': height/diameter per plant × rep (2022:2, 2023:3).

What we do:
  1. Parse to long: hybrid × year × replicate × trait (plot level;
     biometry averaged over plants in a plot after plant QC).
  2. Outlier QC: robustly from mixed-model residuals (year+mother+father+year:rep).
  3. Variance components + repeatability + broad-sense H² (now possible!).
  4. BLUP/adjusted hybrid means -> phenotypes for GWAS/GS.
Output: phase14_reps_pheno/. Old files are left untouched.
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
import warnings, re
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
warnings.filterwarnings("ignore")
ROOT = work()  # undeposited working tree
F=ROOT/"1_данные из вниимка"/"гибриды по повт-тям 2022-2023 итог гг..xlsx"
OUT=ROOT/"phase14_reps_pheno"; OUT.mkdir(exist_ok=True)

def parse_name(s):
    s=str(s).strip()
    m=re.match(r"(ВК101А|ВА761А|ВК934А)\s*[xх×]\s*(ЛИ\d+)",s)
    if m:return en_id(m.group(1)), en_id(m.group(2))
    return None,None

# ---- 1. productivity (yield/oil/seed weight × 3 reps) ----
prod=[]
for year in ["2022","2023"]:
    df=pd.read_excel(F,sheet_name=year,header=None)
    data=df.iloc[2:].reset_index(drop=True)
    for _,row in data.iterrows():
        name=row[9]
        mom,dad=parse_name(name)
        if mom is None:continue
        for rep,(cu,cm,cs) in [(1,(0,1,2)),(2,(3,4,5)),(3,(6,7,8))]:
            prod.append({"mother":mom,"father":dad,"year":int(year),"rep":rep,
                "seed_yield":pd.to_numeric(row[cu],errors="coerce"),
                "oil_content":pd.to_numeric(row[cm],errors="coerce"),
                "seed_weight_1000":pd.to_numeric(row[cs],errors="coerce")})
prod=pd.DataFrame(prod)
print(f"Productivity: {len(prod)} plot-records, hybrids {prod.groupby(['mother','father']).ngroups}")

# ---- biometry (h,d per plant -> plot mean) ----
def parse_biometry(sheet,nrep):
    df=pd.read_excel(F,sheet_name=sheet,header=None)
    # find the name row (contains a cross mark) and the 'h'/'d' row
    name_row=None;hd_row=None
    for r in range(min(8,len(df))):
        vals=[str(v) for v in df.iloc[r].tolist()]
        if any(re.search(r"[xх×]\s*ЛИ\d+",v) for v in vals):name_row=r
        if sum(1 for v in vals if v in ("h","d"))>=4:hd_row=r
    recs=[]
    ncol=df.shape[1]
    # blocks: look for columns that have a name
    namerow=df.iloc[name_row]
    blockcols=[c for c in range(ncol) if parse_name(namerow[c])[0] is not None]
    # each block: name in column c, then nrep (h,d) pairs. Determine the step.
    for c in blockcols:
        mom,dad=parse_name(namerow[c])
        # h/d columns start at c (or c..c+2*nrep-1); locate them from hd_row
        hdvals=df.iloc[hd_row]
        # collect the sequence of h-columns and d-columns from c to the right
        hcols=[];dcols=[]
        cc=c
        while cc<ncol and len(hcols)<nrep:
            if str(hdvals[cc])=="h":hcols.append(cc)
            if str(hdvals[cc])=="d":dcols.append(cc)
            cc+=1
            if cc-c>2*nrep+2:break
        for rep,(hc,dc) in enumerate(zip(hcols,dcols),start=1):
            h=pd.to_numeric(df.iloc[hd_row+1:,hc],errors="coerce").dropna()
            d=pd.to_numeric(df.iloc[hd_row+1:,dc],errors="coerce").dropna()
            recs.append({"mother":mom,"father":dad,"rep":rep,
                "plant_height":h.mean(),"head_diameter":d.mean(),"n_plants":len(h),
                "plant_height_plants":list(h.values),"head_diameter_plants":list(d.values)})
    return pd.DataFrame(recs)

bio=[]
for sheet,nr,yr in [("Биометрия 2022",2,2022),("Биометрия 2023",3,2023)]:
    b=parse_biometry(sheet,nr);b["year"]=yr;bio.append(b)
bio=pd.concat(bio,ignore_index=True)
print(f"Biometry: {len(bio)} plot-records (mean over plants)")

# ---- 2. outlier QC at the plot level (robust mixed-model residuals) ----
removed=[]
def qc_outliers(df,trait,thr=4.0):
    d=df.dropna(subset=[trait]).copy()
    d["g"]=d["mother"]+"_"+d["father"]
    try:
        m=smf.ols(f"Q('{trait}') ~ C(year)+C(mother)+C(father)+C(year):C(rep)",data=d).fit()
        r=m.resid; sr=(r-r.mean())/r.std()
        out=d.index[np.abs(sr)>thr]
    except Exception:
        out=[]
    for i in out:
        removed.append({"trait":trait,"mother":df.loc[i,"mother"],"father":df.loc[i,"father"],
            "year":df.loc[i,"year"],"rep":df.loc[i,"rep"],"value":df.loc[i,trait]})
    df.loc[out,trait]=np.nan
    return len(out)

for t in ["seed_yield","oil_content","seed_weight_1000"]:
    n=qc_outliers(prod,t);print(f"  QC {t}: removed {n} plot-outliers")
for t in ["plant_height","head_diameter"]:
    n=qc_outliers(bio,t);print(f"  QC {t}: removed {n} plot-outliers")
pd.DataFrame(removed).to_csv(OUT/"22_removed_outliers.csv",index=False,encoding="utf-8-sig")

# merge productivity + biometry into one plot-level long table
prod_long=prod.melt(id_vars=["mother","father","year","rep"],value_vars=["seed_yield","oil_content","seed_weight_1000"],
                    var_name="trait",value_name="value")
bio_long=bio.melt(id_vars=["mother","father","year","rep"],value_vars=["plant_height","head_diameter"],
                  var_name="trait",value_name="value")
plot_long=pd.concat([prod_long,bio_long],ignore_index=True).dropna(subset=["value"])
plot_long.to_parquet(OUT/"22_hybrid_plot_level.parquet")

# ---- 3. variance components + repeatability + H² + BLUP ----
import statsmodels.api as sm
rows=[];blups={}
for trait in ["seed_yield","oil_content","seed_weight_1000","plant_height","head_diameter"]:
    d=plot_long[plot_long["trait"]==trait].copy()
    d["g"]=d["mother"]+"_"+d["father"]
    # mixed model: genotype random, year+rep(year) fixed
    try:
        md=smf.mixedlm("value ~ C(year)+C(year):C(rep)",d,groups=d["g"])
        mf=md.fit(reml=True)
        s2g=float(mf.cov_re.iloc[0,0]);s2e=float(mf.scale)
        nrep_avg=d.groupby("g").size().mean()
        H2_mean=s2g/(s2g+s2e/nrep_avg)         # H² on the mean over plots
        repeatability=s2g/(s2g+s2e)            # single-plot repeatability
        # hybrid BLUP = random effect
        re=mf.random_effects
        blup={g:float(v.iloc[0]) for g,v in re.items()}
        # adjusted mean = grand_mean + BLUP
        gm=d["value"].mean()
        blups[trait]={g:gm+b for g,b in blup.items()}
        rows.append({"trait":trait,"σ²g":round(s2g,3),"σ²e":round(s2e,3),
            "plot_repeatability":round(repeatability,3),"H2_of_mean":round(H2_mean,3),
            "n_hybrids":d["g"].nunique(),"n_plots":len(d)})
        print(f"  {trait}: σ²g={s2g:.3f} σ²e={s2e:.3f} repeat={repeatability:.2f} H²(mean)={H2_mean:.2f}")
    except Exception as e:
        print(f"  {trait}: mixedlm error {e}")
vc=pd.DataFrame(rows);vc.to_csv(OUT/"22_variance_components_h2.csv",index=False,encoding="utf-8-sig")

# hybrid BLUP table (wide)
allg=sorted({g for t in blups for g in blups[t]})
bl=pd.DataFrame(index=allg)
for t,dd in blups.items():bl[t]=pd.Series(dd)
bl.index.name="hybrid"
bl["oil_yield"]=bl["seed_yield"]*bl["oil_content"]/100
bl.to_csv(OUT/"22_hybrid_blup_phenotypes.csv",encoding="utf-8-sig")
print(f"\nBLUP phenotypes: {len(bl)} hybrids × {bl.shape[1]} traits -> 22_hybrid_blup_phenotypes.csv")
print(vc.to_string(index=False))
