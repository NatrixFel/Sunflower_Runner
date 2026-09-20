"""
Phase 11 — merge of two VCFs by the user rule.

var1 = bam-merge (466k), var2 = vcf-merge (632k). Both -> 57 canonical samples.
Merge rule (on discrete genotypes 0=hom-ref,1=het,2=hom-alt,-1=missing):
  - position only in one file                    -> take it;
  - one non-missing, the other missing           -> take the non-missing;
  - both non-missing and equal                   -> take it;
  - hom in one, het in the other                 -> take the HOMOZYGOTE;
  - two different homozygotes (0 vs 2)           -> conflict -> missing;
  - both het                                     -> het.
Different REF/ALT at a shared position -> take the var1 variant (bam is more reliable).

Output: phase11_merged_vcf/23_merged.vcf.gz + 23_merged_genotypes.npz (for analysis).
Old files are not touched.
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
import gzip
from pathlib import Path
import numpy as np, allel
ROOT = work()  # undeposited working tree
NEW=ROOT/"6_новые данные_ответы + исходники +vcf"
V1=NEW/"sunflower_var1_bam_merged_15_49.vcf"
V2=NEW/"sunflower_var2_vcf_merged.vcf"
OUT=ROOT/"phase11_merged_vcf"; OUT.mkdir(exist_ok=True)
SEQ={}
for n in range(1,33):SEQ[n]=f"LI{n}"
SEQ[33]="LI34";SEQ[34]="LI35"
for n in range(35,53):SEQ[n]=f"LI{n+2}"
SEQ[53]="LI33";SEQ[54]="LI36";SEQ[55]="VA761";SEQ[56]="VK101";SEQ[57]="VK934"
canon=[SEQ[s] for s in range(1,58)]

def disc_consensus(reps):  # (n,nrep) -> (n,) discrete 0/1/2/-1, hom-priority on a tie
    out=np.full(reps.shape[0],-1,dtype=np.int8)
    for code in (0,2,1):  # priority order: homozygotes, then het
        pass
    # mode with hom priority: count votes
    n=reps.shape[0]
    v0=(reps==0).sum(1);v1=(reps==1).sum(1);v2=(reps==2).sum(1)
    # pick the maximum; on a tie hom (0/2) outranks het(1)
    stack=np.stack([v0,v1,v2],axis=1) # cols: 0,1,2
    mx=stack.max(1)
    has=mx>0
    # priority: if v0==mx -> 0; elif v2==mx ->2; elif v1==mx ->1
    out=np.where(has&(v0==mx),0,out)
    out=np.where(has&(out==-1)&(v2==mx),2,out)
    out=np.where(has&(out==-1)&(v1==mx),1,out)
    return out.astype(np.int8)

def load_disc(path, is_var1):
    print(f"  reading {path.name}...",flush=True)
    cb=allel.read_vcf(str(path),fields=["samples","variants/CHROM","variants/POS","variants/REF","variants/ALT","calldata/GT"])
    ga=allel.GenotypeArray(cb["calldata/GT"]); nalt=ga.to_n_alt(fill=-1).astype(np.int8)
    sr=en_ids(cb["samples"].astype(str)); ci={n:i for i,n in enumerate(sr)}
    disc=np.full((nalt.shape[0],57),-1,dtype=np.int8)
    if is_var1:
        r15=[i for i,s in enumerate(sr) if s.startswith("15-")]
        r49=[i for i,s in enumerate(sr) if s.startswith("49-")]
        c15=disc_consensus(nalt[:,r15]); c49=disc_consensus(nalt[:,r49])
        for j,nm in enumerate(canon):
            if nm=="LI15": disc[:,j]=c15
            elif nm=="LI51": disc[:,j]=c49
            else:
                seq=[k for k,v in SEQ.items() if v==nm][0]; disc[:,j]=nalt[:,ci[f"{seq}.0"]]
    else:
        for j,nm in enumerate(canon):
            seq=[k for k,v in SEQ.items() if v==nm][0]
            # var2 names are integers '1'..'57'
            key=str(seq)
            disc[:,j]=nalt[:,ci[key]] if key in ci else -1
    return cb["variants/CHROM"],cb["variants/POS"].astype(np.int64),cb["variants/REF"],cb["variants/ALT"][:,0],disc

print("Merging VCFs...",flush=True)
c1,p1,ref1,alt1,d1=load_disc(V1,True)
c2,p2,ref2,alt2,d2=load_disc(V2,False)

# keys (chrom,pos): chrom string -> index
chroms=sorted(set(c1)|set(c2)); cidx={c:i for i,c in enumerate(chroms)}
K1=np.array([cidx[c] for c in c1],dtype=np.int64)*10**9+p1
K2=np.array([cidx[c] for c in c2],dtype=np.int64)*10**9+p2
print(f"  var1 {len(K1):,} sites, var2 {len(K2):,}",flush=True)

union=np.union1d(K1,K2)
print(f"  union: {len(union):,} sites",flush=True)
pos1={k:i for i,k in enumerate(K1)}; pos2={k:i for i,k in enumerate(K2)}

def merge_cell(g1,g2):
    # arrays (57,)
    m1=g1<0;m2=g2<0
    out=np.full(57,-1,dtype=np.int8)
    only1=m2&~m1; only2=m1&~m2; both=~m1&~m2
    out=np.where(only1,g1,out); out=np.where(only2,g2,out)
    # both:
    eq=both&(g1==g2)
    out=np.where(eq,g1,out)
    hom1=(g1==0)|(g1==2); hom2=(g2==0)|(g2==2)
    take1=both&~eq&hom1&~hom2; out=np.where(take1,g1,out)
    take2=both&~eq&hom2&~hom1; out=np.where(take2,g2,out)
    bothhet=both&~eq&(g1==1)&(g2==1); out=np.where(bothhet,1,out)
    # conflict of two different homs -> remains -1
    return out

n=len(union)
merged=np.full((n,57),-1,dtype=np.int8)
mref=np.empty(n,dtype=object); malt=np.empty(n,dtype=object)
mchrom=np.empty(n,dtype=object); mpos=np.empty(n,dtype=np.int64)
src=np.zeros(n,dtype=np.int8)  # 1=only var1,2=only var2,3=both
conflict=0
for i,k in enumerate(union):
    i1=pos1.get(k); i2=pos2.get(k)
    ch=chroms[int(k//10**9)]; ps=int(k%10**9)
    mchrom[i]=ch; mpos[i]=ps
    if i1 is not None and i2 is None:
        merged[i]=d1[i1]; mref[i]=ref1[i1]; malt[i]=alt1[i1]; src[i]=1
    elif i2 is not None and i1 is None:
        merged[i]=d2[i2]; mref[i]=ref2[i2]; malt[i]=alt2[i2]; src[i]=2
    else:
        # both
        if ref1[i1]==ref2[i2] and alt1[i1]==alt2[i2]:
            merged[i]=merge_cell(d1[i1],d2[i2]); src[i]=3
        else:
            merged[i]=d1[i1]; conflict+=1; src[i]=3   # different alleles -> var1
        mref[i]=ref1[i1]; malt[i]=alt1[i1]
    if i%100000==0: print(f"    {i:,}/{n:,}",flush=True)

print(f"  sites with different REF/ALT (var1 taken): {conflict}",flush=True)
print(f"  var1 only: {(src==1).sum():,}; var2 only: {(src==2).sum():,}; shared: {(src==3).sum():,}")

# sort by chrom-index, pos
order=np.lexsort((mpos,[cidx[c] for c in mchrom]))
merged=merged[order];mref=mref[order];malt=malt[order];mchrom=mchrom[order];mpos=mpos[order]

np.savez_compressed(OUT/"23_merged_genotypes.npz",
    genotypes=merged,chrom=mchrom.astype(str),pos=mpos,ref=mref.astype(str),alt=malt.astype(str),
    samples=np.array(canon))
print(f"  saved 23_merged_genotypes.npz",flush=True)

# write VCF.gz
gtstr=np.array(["./.","0/0","0/1","1/1"])  # index = g+1
print("  writing 23_merged.vcf.gz...",flush=True)
with gzip.open(OUT/"23_merged.vcf.gz","wt",encoding="utf-8") as f:
    f.write("##fileformat=VCFv4.2\n")
    f.write("##source=merge_var1_var2_hom_priority\n")
    f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
    f.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"+"\t".join(canon)+"\n")
    gi=(merged+1)
    for i in range(len(mpos)):
        gts="\t".join(gtstr[gi[i]])
        f.write(f"{mchrom[i]}\t{mpos[i]}\t.\t{mref[i]}\t{malt[i]}\t.\t.\t.\tGT\t{gts}\n")
print(f"\nDone. merged: {len(mpos):,} sites × 57 samples. Artifacts in {OUT}")
