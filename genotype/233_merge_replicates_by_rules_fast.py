#!/usr/bin/env python3
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
# =============================================================================
# 233_merge_replicates_by_rules_fast.py
# Fast merge of replicate per-sample VCFs into one VCF per sample index.
#
# Reads replicate VCFs from raw_vcf/per_sample/<sample>-<rep>.vcf.gz
# Types (ind/mix) from raw_vcf/per_sample/00_replicate_scheme.tsv
#
# Rules per site:
#   1) missing vs allele -> allele
#   2) homozygous vs heterozygous -> homozygous
#   3) homozygous conflict -> prefer mix over ind over unknown
#   4) remaining tie -> lowest rep wins (deterministic); logged as conflict
#
# Outputs:
#   raw_vcf/per_sample/outputs/merged_vcf/<sample>.vcf.gz
#   raw_vcf/per_sample/outputs/merge_conflicts_fast.tsv
# =============================================================================


import csv
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pysam


@dataclass(frozen=True)
class Rep:
    sample: int
    rep: int
    typ: str  # ind/mix/unknown
    path: str
    name: str  # "<sample>-<rep>"


def find_repo_root() -> str:
    root = calling()
    if not (root / "raw_vcf" / "per_sample").is_dir():
        raise RuntimeError(
            f"SUNFLOWER_CALLING={root} has no raw_vcf/per_sample. "
            "Set it in 05_GitHub/.env."
        )
    return str(root)


def read_scheme(path: str) -> Dict[Tuple[int, int], str]:
    m: Dict[Tuple[int, int], str] = {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            s = int(row["sample"])
            rep = int(row["rep"])
            typ = row["type"].strip().lower()
            m[(s, rep)] = typ
    return m


_re_vcf = re.compile(r"^(\d+)-(\d+)\.vcf\.gz$")


def list_reps(vcf_dir: str, scheme: Dict[Tuple[int, int], str]) -> Dict[int, List[Rep]]:
    reps: Dict[int, List[Rep]] = {}
    for fn in os.listdir(vcf_dir):
        m = _re_vcf.match(fn)
        if not m:
            continue
        s = int(m.group(1))
        rep = int(m.group(2))
        typ = scheme.get((s, rep), "unknown")
        path = os.path.join(vcf_dir, fn)
        name = f"{s}-{rep}"
        reps.setdefault(s, []).append(Rep(sample=s, rep=rep, typ=typ, path=path, name=name))

    # deterministic order: mix first, then ind, then unknown; then rep
    def key(r: Rep) -> Tuple[int, int]:
        rank = 0 if r.typ == "mix" else (1 if r.typ == "ind" else 2)
        return (rank, r.rep)

    for s in reps:
        reps[s].sort(key=key)
    return reps


def gt_is_missing(gt: Tuple[Optional[int], Optional[int]]) -> bool:
    return gt is None or len(gt) != 2 or gt[0] is None or gt[1] is None


def gt_is_hom(gt: Tuple[Optional[int], Optional[int]]) -> bool:
    return (not gt_is_missing(gt)) and gt[0] == gt[1]


def gt_is_het(gt: Tuple[Optional[int], Optional[int]]) -> bool:
    return (not gt_is_missing(gt)) and gt[0] != gt[1]


def type_rank(typ: str) -> int:
    if typ == "mix":
        return 0
    if typ == "ind":
        return 1
    return 2


def choose_gt(gts: List[Tuple[Optional[int], Optional[int]]], reps: List[Rep]) -> Tuple[Tuple[Optional[int], Optional[int]], int, str, bool]:
    """
    Returns: (chosen_gt, chosen_idx, reason, conflict)
    chosen_idx indexes into gts/reps.
    """
    idx_nonmiss = [i for i, gt in enumerate(gts) if not gt_is_missing(gt)]
    if not idx_nonmiss:
        return ((None, None), -1, "all_missing", False)

    # prefer homo if any homo exists
    idx_hom = [i for i in idx_nonmiss if gt_is_hom(gts[i])]
    if idx_hom:
        cand = idx_hom
        reason = "prefer_homo"
    else:
        cand = idx_nonmiss
        reason = "prefer_het"

    # prefer mix over ind over unknown
    best_rank = min(type_rank(reps[i].typ) for i in cand)
    cand2 = [i for i in cand if type_rank(reps[i].typ) == best_rank]
    if best_rank == 0:
        reason += "+prefer_mix"

    # conflict if different GT among best candidates
    gtvals = {(gts[i][0], gts[i][1]) for i in cand2}
    conflict = len(gtvals) > 1
    if conflict:
        reason += "+tie_conflict"

    # deterministic: lowest rep number wins
    chosen = min(cand2, key=lambda i: reps[i].rep)
    return (gts[chosen], chosen, reason, conflict)


def make_single_sample_header(h: pysam.VariantHeader, sample_name: str) -> pysam.VariantHeader:
    """
    pysam headers don't support removing samples in-place reliably across versions.
    Build a new header from textual representation, then add one sample.
    """
    txt = str(h)
    # drop the final #CHROM header line if present
    lines = [ln for ln in txt.splitlines() if not ln.startswith("#CHROM")]
    nh = pysam.VariantHeader()
    for ln in lines:
        if ln.startswith("##"):
            nh.add_line(ln)
    nh.add_sample(sample_name)
    if "GT" not in nh.formats:
        nh.formats.add("GT", 1, "String", "Genotype")
    return nh


def main() -> int:
    root = find_repo_root()
    vcf_dir = os.path.join(root, "raw_vcf", "per_sample")
    out_dir = os.path.join(vcf_dir, "outputs")
    merged_dir = os.path.join(out_dir, "merged_vcf")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    scheme_path = os.path.join(vcf_dir, "00_replicate_scheme.tsv")
    if not os.path.exists(scheme_path):
        raise RuntimeError(f"Missing scheme: {scheme_path}")
    scheme = read_scheme(scheme_path)

    if not shutil_which("bcftools"):
        raise RuntimeError("bcftools not found in PATH")

    rep_map = list_reps(vcf_dir, scheme)
    if not rep_map:
        raise RuntimeError("No <sample>-<rep>.vcf.gz found")

    conflicts_path = os.path.join(out_dir, "merge_conflicts_fast.tsv")
    with open(conflicts_path, "w", newline="") as cf:
        w = csv.writer(cf, delimiter="\t")
        w.writerow(["sample", "n_reps", "n_records", "n_conflict_sites", "note"])

        for sid in sorted(rep_map.keys()):
            reps = rep_map[sid]
            if len(reps) == 1:
                # passthrough copy
                out_gz = os.path.join(merged_dir, f"{sid}.vcf.gz")
                subprocess.check_call(["cp", reps[0].path, out_gz])
                tbi = reps[0].path + ".tbi"
                if os.path.exists(tbi):
                    subprocess.check_call(["cp", tbi, out_gz + ".tbi"])
                w.writerow([sid, 1, "", 0, "passthrough"])
                continue

            # stream bcftools merge as uncompressed BCF to Python
            merge_cmd = ["bcftools", "merge", "-m", "none", "-Ou"] + [r.path for r in reps]
            proc = subprocess.Popen(
                merge_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            assert proc.stdout is not None
            vfin = pysam.VariantFile(proc.stdout)  # BCF stream

            out_path = os.path.join(merged_dir, f"{sid}.vcf.gz")
            hout = make_single_sample_header(vfin.header, str(sid))
            vfout = pysam.VariantFile(out_path, "wz", header=hout)

            sample_names = list(vfin.header.samples)
            # map sample name -> index in reps (names should match "<sample>-<rep>")
            rep_by_name = {r.name: r for r in reps}
            reps_in_order: List[Rep] = []
            for nm in sample_names:
                reps_in_order.append(rep_by_name.get(nm, Rep(sid, 9999, "unknown", "", nm)))

            n_records = 0
            n_conflict = 0
            # streaming BCF: iterate sequentially (no fetch/index)
            for rec in vfin:
                n_records += 1
                alleles = rec.alleles
                if alleles is None or len(alleles) < 2:
                    # monoallelic sites are not informative for this merge; skip
                    continue
                gts = []
                for nm in sample_names:
                    gt = rec.samples[nm].get("GT")
                    if gt is None or len(gt) != 2:
                        gts.append((None, None))
                    else:
                        gts.append((gt[0], gt[1]))

                chosen_gt, chosen_idx, reason, conflict = choose_gt(gts, reps_in_order)
                if conflict:
                    n_conflict += 1

                nrec = vfout.new_record(
                    contig=rec.contig,
                    start=rec.start,
                    stop=rec.stop,
                    id=rec.id,
                    alleles=alleles,
                    qual=rec.qual,
                    filter=list(rec.filter.keys()) if rec.filter.keys() else None,
                )
                # write GT only
                nrec.samples[str(sid)]["GT"] = chosen_gt
                vfout.write(nrec)

            vfout.close()
            vfin.close()
            stderr = proc.stderr.read().decode("utf-8", errors="replace") if proc.stderr else ""
            rc = proc.wait()
            if rc != 0:
                raise RuntimeError(f"bcftools merge failed for {sid}: {stderr[:4000]}")

            # tabix index
            subprocess.check_call(["bcftools", "index", "-t", out_path])
            w.writerow([sid, len(reps), n_records, n_conflict, "ok"])

    print(f"Written:\n- {conflicts_path}\n- merged VCFs in {merged_dir}")
    return 0


def shutil_which(cmd: str) -> Optional[str]:
    from shutil import which
    return which(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
