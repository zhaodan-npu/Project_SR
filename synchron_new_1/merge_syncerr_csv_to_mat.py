#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import savemat

# =========================
# 你无需改任何输入；只需确认目录名与实际一致
# =========================
CSV_DIR = Path("results_syncerr_A1")   # submit_sync_error_A1_array.sh 里 OUTDIR 就是它 :contentReference[oaicite:4]{index=4}
OUT_MAT = Path("syncerrA1_highdim.mat")

# 与 submit_sync_error_A1_array.sh 完全一致的扫描网格 :contentReference[oaicite:5]{index=5}
ALPHAS = np.array([0.0, 0.25, 0.5, 0.75], dtype=float)
TAUS   = np.array([0.0, 0.1, 0.2], dtype=float)
DS     = np.array([i/100.0 for i in range(0, 101)], dtype=float)  # 0.00..1.00 step 0.01

# 路径数（npaths）；你的脚本里是 20 :contentReference[oaicite:6]{index=6}
NPATHS = 20

# 数值取整键，避免浮点比较问题
def key_alpha(x): return float(np.round(x, 3))
def key_tau(x):   return float(np.round(x, 3))
def key_D(x):     return float(np.round(x, 2))

alpha2i = {key_alpha(a): i for i, a in enumerate(ALPHAS)}
tau2i   = {key_tau(t):   i for i, t in enumerate(TAUS)}
D2i     = {key_D(d):     i for i, d in enumerate(DS)}

def main():
    if not CSV_DIR.exists():
        raise FileNotFoundError(f"CSV_DIR not found: {CSV_DIR.resolve()}")

    paths_files = sorted(CSV_DIR.glob("A1syncerr_*_paths.csv"))
    summary_files = sorted(CSV_DIR.glob("A1syncerr_*_summary.csv"))

    if len(paths_files) == 0:
        raise FileNotFoundError(f"No *_paths.csv found under {CSV_DIR.resolve()}")

    NA, NT, ND, M = len(ALPHAS), len(TAUS), len(DS), NPATHS

    # 高维数组：data_Ebar[alpha, tau, D, m]
    data_Ebar   = np.full((NA, NT, ND, M), np.nan, dtype=np.float64)
    data_syncsd = np.full((NA, NT, ND, M), np.nan, dtype=np.float64)  # sqrt(Ebar)

    # 汇总数组：mean/std（如果 summary 缺失，也可由 paths 计算）
    mean_Ebar = np.full((NA, NT, ND), np.nan, dtype=np.float64)
    std_Ebar  = np.full((NA, NT, ND), np.nan, dtype=np.float64)

    # 先读 per-path 文件（最关键）
    filled = 0
    bad = []

    for fp in paths_files:
        try:
            df = pd.read_csv(fp)
            # scan_sync_error_A1.jl 的 per-path 列包含 alpha,tau,D,path,Ebar 等 :contentReference[oaicite:7]{index=7}
            a = key_alpha(float(df["alpha"].iloc[0]))
            t = key_tau(float(df["tau"].iloc[0]))
            d = key_D(float(df["D"].iloc[0]))

            if a not in alpha2i or t not in tau2i or d not in D2i:
                bad.append((fp.name, f"param out of grid: alpha={a}, tau={t}, D={d}"))
                continue

            ia, it, idd = alpha2i[a], tau2i[t], D2i[d]

            # 填 1..NPATHS
            for _, row in df.iterrows():
                m = int(row["path"])  # 1-based
                if 1 <= m <= M:
                    e = float(row["Ebar"])
                    data_Ebar[ia, it, idd, m-1] = e
                    data_syncsd[ia, it, idd, m-1] = np.sqrt(e) if e >= 0 else np.nan

            filled += 1
        except Exception as ex:
            bad.append((fp.name, f"read/parse failed: {ex}"))

    # 再读 summary（可选，用于对照）
    for fp in summary_files:
        try:
            df = pd.read_csv(fp)
            a = key_alpha(float(df["alpha"].iloc[0]))
            t = key_tau(float(df["tau"].iloc[0]))
            d = key_D(float(df["D"].iloc[0]))
            if a not in alpha2i or t not in tau2i or d not in D2i:
                continue
            ia, it, idd = alpha2i[a], tau2i[t], D2i[d]
            mean_Ebar[ia, it, idd] = float(df["mean_Ebar"].iloc[0])
            std_Ebar[ia, it, idd]  = float(df["std_Ebar"].iloc[0])
        except Exception:
            pass

    # 若 summary 没填到，则用 paths 计算补齐
    mask_missing = np.isnan(mean_Ebar)
    if np.any(mask_missing):
        mean_Ebar_calc = np.nanmean(data_Ebar, axis=3)
        std_Ebar_calc  = np.nanstd(data_Ebar, axis=3)
        mean_Ebar[mask_missing] = mean_Ebar_calc[mask_missing]
        std_Ebar[np.isnan(std_Ebar)] = std_Ebar_calc[np.isnan(std_Ebar)]

    # 保存 mat：主键 data 按“高维数组”给出（你可在 Matlab 里直接 load）
    savemat(OUT_MAT, {
        "data": data_syncsd,          # 默认把“同步标准差(=sqrt(Ebar))”作为 data
        "data_Ebar": data_Ebar,       # 同时保存 Ebar
        "mean_Ebar": mean_Ebar,
        "std_Ebar": std_Ebar,
        "alpha_vals": ALPHAS,
        "tau_vals": TAUS,
        "D_vals": DS,
        "npaths": M,
        "order": "data[alpha, tau, D, m] ; m=path index (1..npaths)"
    })

    # 生成缺失报告
    rep = CSV_DIR / "merge_report_syncerr.csv"
    pd.DataFrame({
        "n_paths_csv": [len(paths_files)],
        "n_summary_csv": [len(summary_files)],
        "n_paths_filled": [filled],
        "n_bad": [len(bad)]
    }).to_csv(rep, index=False)

    if bad:
        badrep = CSV_DIR / "merge_bad_files_syncerr.csv"
        pd.DataFrame(bad, columns=["file", "reason"]).to_csv(badrep, index=False)

    print(f"[OK] saved: {OUT_MAT.resolve()}")
    print(f"[INFO] paths_csv={len(paths_files)} summary_csv={len(summary_files)} filled_paths={filled} bad={len(bad)}")
    print(f"[INFO] report: {rep.resolve()}")

if __name__ == "__main__":
    main()
