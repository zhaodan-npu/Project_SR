#!/usr/bin/env julia
# scan_sync_error_A1.jl
#
# 用于：给定 (alpha, tau, D)，做 npaths 次随机路径模拟，
# 计算同步误差 A1: E(t)=mean((x_i - mean(x))^2)，并对稳态区间做时间平均得到 Ebar。
# 输出：两个 CSV（per-path 与 summary），文件名包含 alpha/tau/D。
#
# 【本次修改】
# 1) 三体耦合改为“无显式 1/2”的写法：用 sum_{j<k} 避免 (j,k)/(k,j) 双计数
#    从而将  (1/2)*sum_{j,k}  改写为  sum_{j<k}
# 2) 引入三体张量 B_{ijk}（由 ER 随机图的三角形闭包构造），并按
#       K_i = sum_{j<k} B_{ijk}
#    归一化三体项（当 K_i=0 时三体项取 0）
#
# 你最终的耦合项对应：
#   G_i = α * (σ/K_i) * Σ_{j<k} B_{ijk} tanh(x_j + x_k - 2 x_i)
#       + (1-α) * (σ/N) * Σ_j (x_j - x_i)
#
# 说明：二体项仍保持你原代码的“全连接扩散耦合”形式（与截图一致）。

using Random
using Statistics
using Printf
using DifferentialEquations
using DiffEqCallbacks

# -----------------------------
# 固定系统参数（与截图一致）
# -----------------------------
const a = 1.0
const b = 1.0
const A = 0.2
const Ω = 0.1

const N  = 50
const σ  = 0.1

const dt      = 0.01
const T       = 2000.0
const save_dt = 1.0

# 稳态统计时丢弃前 t_trans 的数据
const t_trans = 500.0

# -----------------------------
# 三体拓扑：ER 图 -> 三角形闭包（simplicial complex）
# -----------------------------
const p_edge   = 0.30          # 你文中随机连边概率
const NET_SEED = 20250101      # 固定拓扑用；如需每个任务不同可改为依赖 alpha/tau/D 的 seed

"""
build_triangles_from_ER(N, p_edge, seed)

返回：
- tri_pairs[i] :: Vector{NTuple{2,Int}}，存所有使 B_{ijk}=1 的 (j,k) 且满足 j<k
- K2[i]        :: Int，等于 length(tri_pairs[i])，即 K_i = Σ_{j<k} B_{ijk}
- n_edges, n_triangles_total（统计信息，方便写入 CSV）
"""
function build_triangles_from_ER(N::Int, p::Float64, seed::Int)
    rng = MersenneTwister(seed)

    # ER 无向图邻接（Bool）
    Aij = falses(N, N)
    n_edges = 0
    for i in 1:N-1
        for j in i+1:N
            if rand(rng) < p
                Aij[i, j] = true
                Aij[j, i] = true
                n_edges += 1
            end
        end
    end

    # 邻居表
    nbrs = Vector{Vector{Int}}(undef, N)
    for i in 1:N
        v = Int[]
        @inbounds for j in 1:N
            if j != i && Aij[i, j]
                push!(v, j)
            end
        end
        nbrs[i] = v
    end

    # 三角形闭包：B_{ijk}=1 当且仅当 (i,j),(i,k),(j,k) 都是边
    tri_pairs = Vector{Vector{NTuple{2,Int}}}(undef, N)
    K2 = Vector{Int}(undef, N)

    for i in 1:N
        pairs = NTuple{2,Int}[]
        nei = nbrs[i]
        L = length(nei)
        for aidx in 1:(L-1)
            j = nei[aidx]
            for bidx in (aidx+1):L
                k = nei[bidx]
                # 强制 j<k 存储，避免双计数
                jj, kk = (j < k) ? (j, k) : (k, j)
                if Aij[jj, kk]  # j-k 也连边 -> 三角形
                    push!(pairs, (jj, kk))
                end
            end
        end
        tri_pairs[i] = pairs
        K2[i] = length(pairs)
    end

    # 统计“全局三角形数”（每个三角形会被 i/j/k 三个节点各记录一次，因此除以 3）
    n_tri_total = round(Int, sum(K2) / 3)

    return tri_pairs, K2, n_edges, n_tri_total
end

# -----------------------------
# 参数结构体
# -----------------------------
mutable struct Params
    D::Float64
    tau::Float64
    sigma::Float64
    alpha::Float64
    N::Int
    tri_pairs::Vector{Vector{NTuple{2,Int}}}
    K2::Vector{Int}
end

# -----------------------------
# 漂移项（含二体+三体混合耦合）
# dx_i = a x_i - b x_i^3 + A sin(Ω t) + G_i + noise_state
# 若 tau>0：引入 η_i 做 OU 色噪声：dη_i = -(η_i/tau) dt + (sqrt(2D)/tau) dW
# -----------------------------
function drift!(du, u, p::Params, t)
    N  = p.N
    α  = p.alpha
    σc = p.sigma
    τ  = p.tau

    x = @view u[1:N]

    # 外驱动
    ext = A * sin(Ω * t)

    # mean-field 二体项需要 sum(x)
    sumx = 0.0
    @inbounds for i in 1:N
        sumx += x[i]
    end

    # 噪声状态（仅 tau>0 时存在）
    η = (τ > 0) ? @view(u[N+1:2N]) : nothing

    @inbounds for i in 1:N
        xi = x[i]

        # ---------- 三体项（无显式 1/2） ----------
        # tri = α * σ / K_i * Σ_{j<k} B_{ijk} tanh(x_j + x_k - 2 x_i)
        tri = 0.0
        if α != 0.0
            Ki = p.K2[i]
            if Ki > 0
                s = 0.0
                xi2 = 2.0 * xi
                for (j, k) in p.tri_pairs[i]
                    s += tanh(x[j] + x[k] - xi2)
                end
                tri = (α * σc / Ki) * s
            end
        end

        # ---------- 二体项（保持你原式：全连接扩散耦合） ----------
        # (1-α) * σ / N * Σ_j (x_j - x_i) = (1-α)*σ/N*(sumx - N*xi)
        pair = ((1.0 - α) * σc / N) * (sumx - N * xi)

        # 噪声输入：白噪声时直接在 diffusion；色噪声时 x 由 η 驱动
        noise_drive = (τ > 0) ? η[i] : 0.0

        du[i] = a * xi - b * xi^3 + ext + tri + pair + noise_drive
    end

    # OU 噪声的漂移项
    if τ > 0
        @inbounds for i in 1:N
            du[N+i] = -η[i] / τ
        end
    end

    return nothing
end

# -----------------------------
# 扩散项（对角噪声）
# tau==0:  dx_i 加白噪声幅度 sqrt(2D)
# tau>0 :  dη_i 加白噪声幅度 sqrt(2D)/tau
# -----------------------------
function diffusion!(dg, u, p::Params, t)
    D  = p.D
    τ  = p.tau
    N  = p.N
    fill!(dg, 0.0)

    if τ == 0.0
        @inbounds for i in 1:N
            dg[i, i] = sqrt(2.0 * D)
        end
    else
        @inbounds for i in 1:N
            dg[N+i, N+i] = sqrt(2.0 * D) / τ
        end
    end
    return nothing
end

# -----------------------------
# 单次路径：返回稳态平均同步误差 Ebar
# E(t)=mean((x_i-mean(x))^2) = mean(x^2) - mean(x)^2
# -----------------------------
function simulate_one_Ebar(seed::Int, p::Params)
    Random.seed!(seed)

    N = p.N
    τ = p.tau

    # 初值：x_i ~ Uniform(-1,1)
    x0 = [2.0 * rand() - 1.0 for _ in 1:N]
    u0 = (τ == 0.0) ? x0 : vcat(x0, zeros(N))

    dim = length(u0)
    noise_rate = zeros(dim, dim)

    prob = SDEProblem(drift!, diffusion!, u0, (0.0, T), p; noise_rate_prototype=noise_rate)

    sv = SavedValues(Float64, Float64)
    save_cb = SavingCallback(
        (u, t, integrator) -> begin
            x = @view u[1:N]
            sumx = 0.0
            sumx2 = 0.0
            @inbounds for i in 1:N
                xi = x[i]
                sumx += xi
                sumx2 += xi * xi
            end
            mx = sumx / N
            (sumx2 / N) - mx * mx
        end,
        sv; saveat = save_dt
    )

    solve(prob, EM(); dt=dt, adaptive=false, callback=save_cb)

    # 稳态平均
    ts = sv.t
    Es = sv.saveval
    idx0 = findfirst(t -> t >= t_trans, ts)
    idx0 === nothing && error("t_trans is larger than T; no steady window.")
    return mean(@view Es[idx0:end])
end

# -----------------------------
# 字符串标签（用于文件名，避免小数点）
# -----------------------------
tag(x::Float64; ndigits::Int=2) = replace(@sprintf("%.*f", ndigits, x), "." => "p")

# -----------------------------
# main
# -----------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 6
        println("Usage: julia --project=. scan_sync_error_A1.jl <alpha> <tau> <D> <npaths> <seedbase> <outdir>")
        exit(1)
    end

    α       = parse(Float64, ARGS[1])
    τ       = parse(Float64, ARGS[2])
    D       = parse(Float64, ARGS[3])
    npaths  = parse(Int,     ARGS[4])
    seedbase= parse(Int,     ARGS[5])
    outdir  = ARGS[6]

    mkpath(outdir)

    # 固定一次拓扑（对所有路径共用）
    tri_pairs, K2, n_edges, n_tri = build_triangles_from_ER(N, p_edge, NET_SEED)

    # 构造参数
    p = Params(D, τ, σ, α, N, tri_pairs, K2)

    # 为保证不同 (α,τ,D) 的 seed 不冲突，做一个可重复的组合偏移
    αtag = round(Int, α * 1000)   # 0.25 -> 250
    τtag = round(Int, τ * 1000)   # 0.1  -> 100
    Dtag = round(Int, D * 1000)   # 0.05 -> 50

    Eb = Vector{Float64}(undef, npaths)
    seeds = Vector{Int}(undef, npaths)

    for pidx in 1:npaths
        seed = seedbase + αtag * 1_000_000 + τtag * 1_000 + Dtag * 10 + pidx
        seeds[pidx] = seed
        Eb[pidx] = simulate_one_Ebar(seed, p)
    end

    meanE = mean(Eb)
    stdE  = std(Eb)

    # 输出文件
    fname_base = "A1syncerr_alpha$(tag(α;ndigits=2))_tau$(tag(τ;ndigits=2))_D$(tag(D;ndigits=2))"
    f_paths   = joinpath(outdir, fname_base * "_paths.csv")
    f_summary = joinpath(outdir, fname_base * "_summary.csv")

    # per-path
    open(f_paths, "w") do io
        println(io, "alpha,tau,D,path,seed,Ebar,t_trans,T,dt,p_edge,net_seed,n_edges,n_triangles")
        for i in 1:npaths
            println(io, @sprintf("%.6f,%.6f,%.6f,%d,%d,%.10e,%.2f,%.2f,%.4f,%.2f,%d,%d,%d",
                                 α, τ, D, i, seeds[i], Eb[i], t_trans, T, dt,
                                 p_edge, NET_SEED, n_edges, n_tri))
        end
    end

    # summary
    open(f_summary, "w") do io
        println(io, "alpha,tau,D,npaths,mean_Ebar,std_Ebar,t_trans,T,dt,p_edge,net_seed,n_edges,n_triangles")
        println(io, @sprintf("%.6f,%.6f,%.6f,%d,%.10e,%.10e,%.2f,%.2f,%.4f,%.2f,%d,%d,%d",
                             α, τ, D, npaths, meanE, stdE, t_trans, T, dt,
                             p_edge, NET_SEED, n_edges, n_tri))
    end

    @info "Done" alpha=α tau=τ D=D npaths=npaths meanE=meanE stdE=stdE outdir=outdir p_edge=p_edge net_seed=NET_SEED n_edges=n_edges n_triangles=n_tri
end
