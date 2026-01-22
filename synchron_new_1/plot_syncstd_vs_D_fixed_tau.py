#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=str, required=True,
                    help="results folder containing *_paths.csv, e.g. results_sync_A1")
    ap.add_argument("--tau", type=float, required=True,
                    help="fixed tau value, e.g. 0.10")
    ap.add_argument("--outdir", type=str, default="figs_sync_A1")
    ap.add_argument("--with_errorbar", action="store_true",
                    help="plot ±1 std over paths")
    ap.add_argument("--fmt", type=str, default="png,pdf",
                    help="output formats, e.g. png or png,pdf")
    args = ap.parse_args()

    indir = Path(args.indir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(indir / "*_paths.csv")))
    if not files:
        raise FileNotFoundError(f"No *_paths.csv found under {indir.resolve()}")

    # 读取所有 paths
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    # 必要列检查（scan_sync_error_A1.jl 输出里应有 Ebar/alpha/tau/D）:contentReference[oaicite:1]{index=1}
    need = ["alpha", "tau", "D", "Ebar"]
    for c in need:
        if c not in df.columns:
            raise KeyError(f"Missing column '{c}'. Available columns: {list(df.columns)}")

    # 数值化
    for c in ["alpha", "tau", "D", "Ebar"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 固定 tau
    tau0 = args.tau
    sub = df[np.isclose(df["tau"].values, tau0, rtol=0, atol=1e-12)].copy()
    if sub.empty:
        taus = np.sort(df["tau"].dropna().unique())
        raise ValueError(f"No rows found for tau={tau0}. Available taus: {taus}")

    # 同步标准差：sqrt(Ebar)
    sub["sync_std"] = np.sqrt(sub["Ebar"].clip(lower=0.0))

    # 按 (alpha, D) 聚合：均值 + 标准差（路径间）
    g = sub.groupby(["alpha", "D"], as_index=False)["sync_std"].agg(["mean", "std"]).reset_index()
    g = g.rename(columns={"mean": "mean_sync_std", "std": "std_sync_std"})

    # 绘图风格（不指定颜色，matplotlib 自动配色）
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

    alphas = np.sort(g["alpha"].unique())

    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111)

    for a in alphas:
        s = g[np.isclose(g["alpha"].values, a, rtol=0, atol=1e-12)].sort_values("D")
        x = s["D"].values
        y = s["mean_sync_std"].values

        if args.with_errorbar:
            yerr = s["std_sync_std"].values
            ax.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.6, capsize=3,
                        label=fr"$\alpha={a:.2f}$")
        else:
            ax.plot(x, y, marker="o", linewidth=1.6, label=fr"$\alpha={a:.2f}$")

    ax.set_xlabel(r"Noise intensity $D$")
    ax.set_ylabel(r"Mean synchronization std  $\langle \sqrt{\overline{E}} \rangle$")
    ax.set_title(fr"Mean sync-std vs $D$ at fixed $\tau={tau0:.2f}$")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()

    fmts = [x.strip() for x in args.fmt.split(",") if x.strip()]
    base = outdir / f"syncstd_vs_D_tau{tau0:.2f}"
    for f in fmts:
        fig.savefig(str(base) + f".{f}")
    plt.close(fig)

    print(f"[OK] saved: {base}.[{','.join(fmts)}]")


if __name__ == "__main__":
    main()
