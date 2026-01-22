#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_fpt_A1_to_mat.py

Merge results_fpt_A1/*_{summary,paths}.csv into a MATLAB .mat file.

Usage:
  python merge_fpt_A1_to_mat.py --indir results_fpt_A1 --out merged_fpt_A1.mat --with-paths

Outputs (.mat variables):
  alpha_vals, tau_vals, D_vals
  hitrate, nhit, npaths
  mpft_hit, std_hit, mpft_cens, std_cens
  missing_mask
  (optional) t_fpt, t_fpt_cens, hitflag, seed
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import savemat


PAT = re.compile(
    r"A1fpt_alpha(?P<alpha>[0-9p]+)_tau(?P<tau>[0-9p]+)_D(?P<D>[0-9p]+)_(?P<kind>summary|paths)\.csv$"
)


def p2f(s: str) -> float:
    # "0p25" -> 0.25
    return float(s.replace("p", "."))


def key100(x: float) -> int:
    # consistent indexing with 2-decimal grid
    return int(round(x * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Input dir, e.g., results_fpt_A1")
    ap.add_argument("--out", required=True, help="Output .mat path")
    ap.add_argument("--with-paths", action="store_true", help="Also merge *_paths.csv into 4D arrays")
    args = ap.parse_args()

    indir = Path(args.indir)
    out = Path(args.out)

    alpha_vals = np.array([0.00, 0.25, 0.50, 0.75, 1.00], dtype=float)
    tau_vals   = np.array([0.00, 0.10, 0.20], dtype=float)
    D_vals     = np.round(np.linspace(0.00, 1.00, 51), 2)

    NA, NT, ND = len(alpha_vals), len(tau_vals), len(D_vals)

    idxA = {key100(v): i for i, v in enumerate(alpha_vals)}
    idxT = {key100(v): i for i, v in enumerate(tau_vals)}
    idxD = {key100(v): i for i, v in enumerate(D_vals)}

    # 3D summary arrays
    hitrate    = np.full((NA, NT, ND), np.nan, dtype=float)
    mpft_hit   = np.full((NA, NT, ND), np.nan, dtype=float)
    std_hit    = np.full((NA, NT, ND), np.nan, dtype=float)
    mpft_cens  = np.full((NA, NT, ND), np.nan, dtype=float)
    std_cens   = np.full((NA, NT, ND), np.nan, dtype=float)
    nhit       = np.full((NA, NT, ND), -1, dtype=np.int32)
    npaths_arr = np.full((NA, NT, ND), -1, dtype=np.int32)
    missing    = np.ones((NA, NT, ND), dtype=bool)

    # collect summary files
    summary_files = sorted(indir.glob("*_summary.csv"))
    if not summary_files:
        raise FileNotFoundError(f"No *_summary.csv in {indir}")

    # infer NPATHS from the first summary file
    d0 = pd.read_csv(summary_files[0])
    if "npaths" not in d0.columns:
        raise ValueError(f"Missing column 'npaths' in {summary_files[0]}")
    NPATHS = int(d0.loc[0, "npaths"])

    # optional 4D arrays
    t_fpt = t_fpt_cens = hitflag = seed = None
    if args.with_paths:
        t_fpt      = np.full((NPATHS, NA, NT, ND), np.nan, dtype=float)
        t_fpt_cens = np.full((NPATHS, NA, NT, ND), np.nan, dtype=float)
        hitflag    = np.full((NPATHS, NA, NT, ND), -1, dtype=np.int8)
        seed       = np.full((NPATHS, NA, NT, ND), -1, dtype=np.int32)

    merged = 0
    for sf in summary_files:
        m = PAT.match(sf.name)
        if not m:
            continue

        a = p2f(m.group("alpha"))
        t = p2f(m.group("tau"))
        d = p2f(m.group("D"))

        ka, kt, kd = key100(a), key100(t), key100(d)
        if ka not in idxA or kt not in idxT or kd not in idxD:
            continue

        ia, it, id_ = idxA[ka], idxT[kt], idxD[kd]

        df = pd.read_csv(sf)
        if len(df) != 1:
            raise ValueError(f"Expected 1-row summary CSV, got {len(df)} rows: {sf}")

        npaths_here = int(df.loc[0, "npaths"])
        if npaths_here != NPATHS:
            raise ValueError(f"NPATHS mismatch: {sf} has {npaths_here}, expected {NPATHS}")

        npaths_arr[ia, it, id_] = npaths_here
        nhit[ia, it, id_]       = int(df.loc[0, "nhit"])
        hitrate[ia, it, id_]    = float(df.loc[0, "hitrate"])
        mpft_hit[ia, it, id_]   = float(df.loc[0, "mpft_hit"])
        std_hit[ia, it, id_]    = float(df.loc[0, "std_hit"])
        mpft_cens[ia, it, id_]  = float(df.loc[0, "mpft_cens"])
        std_cens[ia, it, id_]   = float(df.loc[0, "std_cens"])
        missing[ia, it, id_]    = False
        merged += 1

        if args.with_paths:
            pf = sf.with_name(sf.name.replace("_summary.csv", "_paths.csv"))
            if not pf.exists():
                # leave as missing for paths
                continue
            dpf = pd.read_csv(pf)

            # required columns
            req = ["path", "seed", "hit", "t_fpt", "t_fpt_censored"]
            for col in req:
                if col not in dpf.columns:
                    raise ValueError(f"Missing column '{col}' in {pf}")

            # path is 1..NPATHS in your Julia output
            pidx = dpf["path"].to_numpy(dtype=int) - 1
            ok = (pidx >= 0) & (pidx < NPATHS)
            pidx = pidx[ok]

            t_fpt[pidx, ia, it, id_]      = dpf.loc[ok, "t_fpt"].to_numpy(dtype=float)
            t_fpt_cens[pidx, ia, it, id_] = dpf.loc[ok, "t_fpt_censored"].to_numpy(dtype=float)
            hitflag[pidx, ia, it, id_]    = dpf.loc[ok, "hit"].to_numpy(dtype=np.int8)
            seed[pidx, ia, it, id_]       = dpf.loc[ok, "seed"].to_numpy(dtype=np.int32)

    expected = NA * NT * ND
    print(f"[merge] summaries merged={merged} (expected {expected})")
    if merged != expected:
        miss_n = int(missing.sum())
        print(f"[warn] missing combos: {miss_n}. Check failed/absent tasks.")

    mat = {
        "alpha_vals": alpha_vals,
        "tau_vals": tau_vals,
        "D_vals": D_vals,
        "hitrate": hitrate,
        "nhit": nhit,
        "npaths": npaths_arr,
        "mpft_hit": mpft_hit,
        "std_hit": std_hit,
        "mpft_cens": mpft_cens,
        "std_cens": std_cens,
        "missing_mask": missing,
    }

    if args.with_paths:
        mat.update({
            "t_fpt": t_fpt,
            "t_fpt_cens": t_fpt_cens,
            "hitflag": hitflag,
            "seed": seed,
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(out, mat, do_compression=True)
    print(f"[merge] wrote: {out}")


if __name__ == "__main__":
    main()
