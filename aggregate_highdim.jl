#!/usr/bin/env julia
using MAT

# 合并已跑好的单文件，生成完整的 batch_all.mat（高维数组覆盖全部参数）
# 参数与 submit_array.sh 保持一致：alpha × tau × D × path

# --------------- 可调参数 ---------------
alpha_vals = [0.0, 0.25, 0.5, 0.75]
tau_vals   = [0.0, 0.1, 0.2]
D_vals     = collect(0.0:0.02:1.0)   # 51 个
NUM_PATHS  = 20                      # path = 0..19
time_points = 2001                   # 0:1:2000，共 2001 点

# 输出目录；默认当前目录，可通过命令行第一个参数覆盖
outdir = length(ARGS) >= 1 ? ARGS[1] : pwd()

# --------------- 预分配高维数组 ---------------
A = length(alpha_vals); T = length(tau_vals); Dn = length(D_vals); P = NUM_PATHS
mean_all  = fill(NaN, A, T, Dn, P, time_points)
sync_all  = fill(NaN, A, T, Dn, P, time_points)
Ebar_all  = fill(NaN, A, T, Dn, P)
hit_all   = fill(0,   A, T, Dn, P)
fpt_all   = fill(NaN, A, T, Dn, P)
fptc_all  = fill(NaN, A, T, Dn, P)

times_ref = nothing
meta_ref  = nothing

# --------------- 主循环：逐文件填充 ---------------
for (ai, alpha) in enumerate(alpha_vals),
    (ti, tau)   in enumerate(tau_vals),
    (di, Dval)  in enumerate(D_vals),
    path_idx in 0:NUM_PATHS-1

    fname = joinpath(outdir, "alpha$(alpha)_tau$(tau)_D$(Dval)_path$(path_idx).mat")
    if !isfile(fname)
        @warn "文件不存在，填充 NaN" fname
        continue
    end

    data = matread(fname)

    # 读取时间轴
    if times_ref === nothing && haskey(data, "times")
        times_ref = vec(data["times"])
    end

    # 读取 meta（只取第一个文件的参数信息）
    if meta_ref === nothing && haskey(data, "params")
        meta_ref = data["params"]
    end

    # 提取并填充各指标
    if haskey(data, "mean")
        mean_all[ai, ti, di, path_idx + 1, :] = vec(data["mean"])
    end
    if haskey(data, "sync")
        sync_all[ai, ti, di, path_idx + 1, :] = vec(data["sync"])
    end
    Ebar_all[ai, ti, di, path_idx + 1] = get(data, "Ebar", NaN)
    hit_all[ai, ti, di, path_idx + 1]  = get(data, "hit", 0)
    fpt_all[ai, ti, di, path_idx + 1]  = get(data, "t_fpt", NaN)
    fptc_all[ai, ti, di, path_idx + 1] = get(data, "t_fpt_censored", NaN)
end

# 兜底时间轴
times_ref = times_ref === nothing ? collect(0:time_points-1) : times_ref

# --------------- 写出 batch_all.mat ---------------
mkpath(outdir)
batch_file = joinpath(outdir, "batch_all.mat")
matwrite(batch_file, Dict(
    "alphas" => alpha_vals,
    "taus"   => tau_vals,
    "Ds"     => D_vals,
    "paths"  => collect(0:NUM_PATHS-1),
    "times"  => times_ref,
    "mean_all" => mean_all,
    "sync_all" => sync_all,
    "Ebar_all" => Ebar_all,
    "hit_all"  => hit_all,
    "t_fpt_all" => fpt_all,
    "t_fpt_censored_all" => fptc_all,
    "meta" => meta_ref,
))

println("batch_all.mat 已生成：", batch_file)
