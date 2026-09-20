"""Local paths for the GitHub code tree.

Committed defaults point at the deposit next to this folder. Machine-specific
roots (raw VCF/BAM, primary Excel, ChoCallate, reference FASTA) come from a
gitignored `.env` — copy `.env.example` and fill the blanks.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load_env() -> None:
    env = HERE / ".env"
    if not env.exists():
        return
    for raw in env.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_env()


def _env_path(name: str, default: Path | None = None) -> Path:
    raw = os.environ.get(name, "").strip()
    if raw:
        return Path(raw).expanduser()
    if default is not None:
        return default
    raise RuntimeError(
        f"Set {name} in {HERE / '.env'} (copy .env.example). "
        "That file is gitignored."
    )


# Deposit call sets (VCF / NPZ) uploaded with the paper.
DEPOSIT = _env_path(
    "SUNFLOWER_DEPOSIT",
    default=(HERE.parent / "04_Депозит"),
)

# Tables written by scripts (CSV / parquet / JSON / GAPIT).
OUT = _env_path("SUNFLOWER_OUT", default=(HERE / "outputs"))
OUT.mkdir(parents=True, exist_ok=True)
INTERMEDIATE = OUT

# Undeposited inputs: primary Excel, var1/var2 VCF. Optional until a script reads them.
_work = os.environ.get("SUNFLOWER_WORK", "").strip()
WORK = Path(_work).expanduser() if _work else None

# Calling-pipeline root (raw_vcf/, raw_vcf_merged/, merged BAM). Optional.
_calling = os.environ.get("SUNFLOWER_CALLING", "").strip()
CALLING = Path(_calling).expanduser() if _calling else None

# ChoCallate checkout and the sunflower reference. Optional.
_cho = os.environ.get("CHOCALLATE", "").strip()
CHOCALLATE = Path(_cho).expanduser() if _cho else None
_ref = os.environ.get("SUNFLOWER_REF", "").strip()
REF = Path(_ref).expanduser() if _ref else None

from names import (  # re-export so scripts can import from paths
    en_table,
    en_ids,
    en_id,
    en_trait,
    ru_id,
    ru_col,
    ru_trait,
    is_line_id,
)

MODELS = HERE / "scripts" / "models"


def use_models() -> None:
    p = str(MODELS)
    if p not in sys.path:
        sys.path.insert(0, p)


def work() -> Path:
    if WORK is None:
        raise RuntimeError("Set SUNFLOWER_WORK in .env for undeposited inputs.")
    return WORK


def calling() -> Path:
    if CALLING is None:
        raise RuntimeError("Set SUNFLOWER_CALLING in .env for the calling tree.")
    return CALLING


# Compatibility aliases used by older scripts that said ROOT/DATA/D for the deposit.
ROOT = DEPOSIT
DATA = DEPOSIT
D = DEPOSIT
