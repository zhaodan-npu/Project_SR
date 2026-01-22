#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def pick_metric_column(df, preferred):
    """优先用用户指定列；若不存在，则自动回退到常见列名。"""
    if preferred in df.columns:
        return preferred
    candidates = [
        "mpft_hit", "mfpt", "MFPT", "mpft", "mean_fpt", "mean_FPT",
        "mpft_cens", "mpft_censored"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"MFPT column not found. Available columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, default="results_fpt_A1",
                    help="folder containing *_summary.csv")
    ap.add_argument("--tau", type=float, required=True,
                    help="fixed tau value, e.g. 0.10")
    ap.add_argument("--metric", type=str, default="mpft_hit",
                    help="MFPT column name (default mpft_hit). Auto-fallback if missing.")
    ap.add_argument("--outdir", type=str, default="figs_fpt_A1")
    ap.add_argument("--fmt", type=str, default="png,pdf",
                    help="output formats, e.g. png or png,pdf")
    ap.add_argument("--min_hitrate", type=float, default=None,
                    help="optional: mask points with hitrate < threshold")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(indir / "*_summary.csv")))
    if not files:
        raise FileNotFoundError(f"No *_summary.csv found under {indir.resolve()}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # basic numeric conversion
    for c in ["alpha", "tau", "D"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    metric = pick_metric_column(df, args.metric)

    # filter fixed tau (tolerance handles float writing)
    tau0 = args.tau
    sub = df[np.isclose(df["tau"].values, tau0, rtol=0, atol=1e-12)].copy()
    if sub.empty:
        taus = np.sort(df["tau"].dropna().unique())
        raise ValueError(f"No rows found for tau={tau0}. Available taus: {taus}")

    # if multiple rows per (alpha,D), average them
    agg_cols = [metric]
    if "std_hit" in sub.columns:
        # 可选：用于误差条（如果你想画 errorbar）
        pass
    if "hitrate" in sub.columns:
        agg_cols.append("hitrate")

    sub = sub.groupby(["alpha", "D"], as_index=False)[agg_cols].mean()

    # optional mask by hitrate
    if args.min_hitrate is not None and "hitrate" in sub.columns:
        sub.loc[sub["hitrate"] < args.min_hitrate, metric] = np.nan

    # plotting style
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "Times New Roman",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
    })

    alphas = np.sort(sub["alpha"].unique())
    Ds_all = np.sort(sub["D"].unique())

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)

    for a in alphas:
        s = sub[np.isclose(sub["alpha"].values, a, rtol=0, atol=1e-12)].copy()
        s = s.sort_values("D")
        ax.plot(s["D"].values, s[metric].values, marker="o", linewidth=1.6,
                label=fr"$\alpha={a:.2f}$")

    ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(f"{metric}")
    ax.set_title(fr"MFPT vs $D$ at fixed $\tau={tau0:.2f}$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()

    fmts = [x.strip() for x in args.fmt.split(",") if x.strip()]
    base = outdir / f"MFPT_vs_D_tau{tau0:.2f}_{metric}"
    for f in fmts:
        fig.savefig(str(base) + f".{f}")

    plt.close(fig)
    print(f"[OK] saved: {base}.[{','.join(fmts)}]")
    if args.min_hitrate is not None:
        print(f"[INFO] masked points with hitrate < {args.min_hitrate}")


if __name__ == "__main__":
    main()
