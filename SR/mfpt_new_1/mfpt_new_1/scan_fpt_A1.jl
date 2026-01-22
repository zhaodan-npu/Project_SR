#!/usr/bin/env julia
# scan_fpt_A1.jl
#
# 【最终版】方程完全对齐 scan_sync_error_A1.jl：
#   G_i = α * (σ/K_i) * Σ_{j<k} B_{ijk} tanh(x_j + x_k - 2 x_i)
#       + (1-α) * (σ/N) * Σ_j (x_j - x_i)
# 其中 B_{ijk} 由 ER 图的三角形闭包构造；K_i = Σ_{j<k} B_{ijk}；若 K_i=0 则三体项为 0。
#
# FPT 定义：
#   FPT = inf{ t>=0 : mean(x(t)) >= X_THR }
# 默认 X_INIT=-1，X_THR=0（从左井首次越过势垒）
#
# 用法：
#   julia --project=. scan_fpt_A1.jl <alpha> <tau> <D> <npaths> <seedbase> <outdir>
#
# 输出：
#   *_paths.csv   每条路径的 hit 与 t_fpt
#   *_summary.csv 汇总：hitrate、mpft_hit/std_hit、mpft_cens/std_cens

using Random
using Statistics
using Printf
using DifferentialEquations
using DiffEqCallbacks

# -----------------------------
# 固定系统参数（与 scan_sync_error_A1.jl 一致）:contentReference[oaicite:1]{index=1}
# -----------------------------
const a = 1.0
const b = 1.0
const A = 0.2
const Ω = 0.1

const N  = 50
const σ  = 0.1

const dt   = 0.01
const TMAX = 200.0          # 首穿时间窗口（你 MATLAB 例子用 200）

# -----------------------------
# 首穿定义（可按需修改）
# -----------------------------
const X_INIT = -1.0         # 初值：全体在左井
const X_THR  = 0.0          # 首次达到阈值（默认越过势垒 0）

# -----------------------------
# 三体拓扑：ER 图 -> 三角形闭包（与 scan_sync_error_A1.jl 一致）:contentReference[oaicite:2]{index=2}
# -----------------------------
const p_edge   = 0.30
const NET_SEED = 20250101   # 固定拓扑（与 sync 一致）

"""
build_triangles_from_ER(N, p_edge, seed)

返回：
- tri_pairs[i] :: Vector{NTuple{2,Int}}，存所有使 B_{ijk}=1 的 (j,k) 且满足 j<k
- K2[i]        :: Int，K_i = length(tri_pairs[i])
- n_edges, n_triangles_total（统计信息）
"""
function build_triangles_from_ER(N::Int, p::Float64, seed::Int)
    rng = MersenneTwister(seed)

    # ER 无向图邻接
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
                jj, kk = (j < k) ? (j, k) : (k, j)   # 强制 j<k 存储，避免双计数
                if Aij[jj, kk]
                    push!(pairs, (jj, kk))
                end
            end
        end
        tri_pairs[i] = pairs
        K2[i] = length(pairs)
    end

    # 全局三角形数：每个三角形会被 i/j/k 记录一次，所以除以 3
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
# mean(x)（用于首穿判据）
# -----------------------------
@inline function mean_x(u, N::Int)
    s = 0.0
    @inbounds for i in 1:N
        s += u[i]
    end
    return s / N
end

# -----------------------------
# 漂移项（完全对齐 scan_sync_error_A1.jl）:contentReference[oaicite:3]{index=3}
# -----------------------------
function drift!(du, u, p::Params, t)
    N  = p.N
    α  = p.alpha
    σc = p.sigma
    τ  = p.tau

    x = @view u[1:N]
    ext = A * sin(Ω * t)

    # 二体项：需要 sum(x)
    sumx = 0.0
    @inbounds for i in 1:N
        sumx += x[i]
    end

    # 噪声状态（tau>0 时）
    η = (τ > 0) ? @view(u[N+1:2N]) : nothing

    @inbounds for i in 1:N
        xi = x[i]

        # 三体项（无显式 1/2）：α * σ/K_i * Σ_{j<k} B_{ijk} tanh(x_j + x_k - 2 x_i)
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

        # 二体扩散耦合： (1-α)*σ/N * Σ_j (x_j - x_i)
        pair = ((1.0 - α) * σc / N) * (sumx - N * xi)

        # 色噪声时：x 由 η 驱动；白噪声在 diffusion 中
        noise_drive = (τ > 0) ? η[i] : 0.0

        du[i] = a * xi - b * xi^3 + ext + tri + pair + noise_drive
    end

    # OU 噪声漂移
    if τ > 0
        @inbounds for i in 1:N
            du[N+i] = -η[i] / τ
        end
    end

    return nothing
end

# -----------------------------
# 扩散项（与 scan_sync_error_A1.jl 一致）:contentReference[oaicite:4]{index=4}
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
# 单次路径：FPT（离散步点时间，不做插值，便于与 MATLAB 一致）
# 返回：hit::Bool, t_fpt::Float64, t_fpt_cens::Float64
# -----------------------------
function simulate_one_fpt(seed::Int, p::Params)
    Random.seed!(seed)

    N = p.N
    τ = p.tau

    # 初值：全体在左井
    x0 = fill(X_INIT, N)
    u0 = (τ == 0.0) ? x0 : vcat(x0, zeros(N))

    dim = length(u0)
    noise_rate = zeros(dim, dim)

    prob = SDEProblem(drift!, diffusion!, u0, (0.0, TMAX), p; noise_rate_prototype=noise_rate)

    hit = Ref(false)
    t_hit = Ref(NaN)

    # 每步检查 mean(x) 是否越过阈值
    condition(u, t, integrator) = (mean_x(u, N) >= X_THR)

    function affect!(integrator)
        hit[] = true
        t_hit[] = integrator.t   # 不插值：与离散步统计一致
        terminate!(integrator)
    end

    cb = DiscreteCallback(condition, affect!; save_positions=(false, false))

    solve(prob, EM(); dt=dt, adaptive=false, callback=cb, save_everystep=false)

    if hit[]
        return true, t_hit[], t_hit[]
    else
        return false, NaN, TMAX
    end
end

# -----------------------------
# 文件名 tag（避免小数点）
# -----------------------------
tag(x::Float64; ndigits::Int=2) = replace(@sprintf("%.*f", ndigits, x), "." => "p")

# -----------------------------
# main
# -----------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 6
        println("Usage: julia --project=. scan_fpt_A1.jl <alpha> <tau> <D> <npaths> <seedbase> <outdir>")
        exit(1)
    end

    α       = parse(Float64, ARGS[1])
    τ       = parse(Float64, ARGS[2])
    D       = parse(Float64, ARGS[3])
    npaths  = parse(Int,     ARGS[4])
    seedbase= parse(Int,     ARGS[5])
    outdir  = ARGS[6]

    mkpath(outdir)

    # 固定一次拓扑（对所有路径共用）:contentReference[oaicite:5]{index=5}
    tri_pairs, K2, n_edges, n_tri = build_triangles_from_ER(N, p_edge, NET_SEED)

    # 参数
    p = Params(D, τ, σ, α, N, tri_pairs, K2)

    # 与 sync 一致的 seed 偏移策略
    αtag = round(Int, α * 1000)
    τtag = round(Int, τ * 1000)
    Dtag = round(Int, D * 1000)

    tfpt = Vector{Float64}(undef, npaths)
    tfpt_cens = Vector{Float64}(undef, npaths)
    hitflag = Vector{Int}(undef, npaths)
    seeds = Vector{Int}(undef, npaths)

    for pidx in 1:npaths
        seed = seedbase + αtag * 1_000_000 + τtag * 1_000 + Dtag * 10 + pidx
        seeds[pidx] = seed

        hit, t1, tc = simulate_one_fpt(seed, p)
        hitflag[pidx] = hit ? 1 : 0
        tfpt[pidx] = t1
        tfpt_cens[pidx] = tc
    end

    nhit = sum(hitflag)
    hitrate = nhit / npaths

    # MPFT：只对命中样本平均
    mpft = (nhit > 0) ? mean(tfpt[hitflag .== 1]) : NaN
    spft = (nhit > 1) ? std(tfpt[hitflag .== 1])  : NaN

    # 删失平均：未命中按 TMAX 计入
    mpft_cens = mean(tfpt_cens)
    spft_cens = std(tfpt_cens)

    fname_base = "A1fpt_alpha$(tag(α;ndigits=2))_tau$(tag(τ;ndigits=2))_D$(tag(D;ndigits=2))"
    f_paths   = joinpath(outdir, fname_base * "_paths.csv")
    f_summary = joinpath(outdir, fname_base * "_summary.csv")

    # per-path
    open(f_paths, "w") do io
        println(io, "alpha,tau,D,path,seed,hit,t_fpt,t_fpt_censored,TMAX,dt,x_init,x_thr,p_edge,net_seed,n_edges,n_triangles")
        for i in 1:npaths
            println(io, @sprintf("%.6f,%.6f,%.6f,%d,%d,%d,%.10e,%.10e,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d",
                                 α, τ, D, i, seeds[i], hitflag[i], tfpt[i], tfpt_cens[i],
                                 TMAX, dt, X_INIT, X_THR,
                                 p_edge, NET_SEED, n_edges, n_tri))
        end
    end

    # summary
    open(f_summary, "w") do io
        println(io, "alpha,tau,D,npaths,nhit,hitrate,mpft_hit,std_hit,mpft_cens,std_cens,TMAX,dt,x_init,x_thr,p_edge,net_seed,n_edges,n_triangles")
        println(io, @sprintf("%.6f,%.6f,%.6f,%d,%d,%.6f,%.10e,%.10e,%.10e,%.10e,%.2f,%.4f,%.2f,%.2f,%.2f,%d,%d,%d",
                             α, τ, D, npaths, nhit, hitrate, mpft, spft, mpft_cens, spft_cens,
                             TMAX, dt, X_INIT, X_THR,
                             p_edge, NET_SEED, n_edges, n_tri))
    end

    @info "Done" alpha=α tau=τ D=D npaths=npaths nhit=nhit hitrate=hitrate mpft=mpft mpft_cens=mpft_cens outdir=outdir p_edge=p_edge net_seed=NET_SEED n_edges=n_edges n_triangles=n_tri
end
