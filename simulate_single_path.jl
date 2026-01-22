#!/usr/bin/env julia
using Random, Distributions, Statistics, DifferentialEquations, DiffEqCallbacks, MAT

# -----------------------------
# 模型参数（与 synchron_new_1 保持一致）
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

# 三体拓扑：ER 图 + 三角形闭包
const p_edge   = 0.30
const NET_SEED = 20250101

# -----------------------------
# 三角形张量生成（摘自 synchron_new_1）
# -----------------------------
"""
build_triangles_from_ER(N, p_edge, seed)

返回：
- tri_pairs[i] :: Vector{NTuple{2,Int}}，存所有使 B_{ijk}=1 的 (j,k) 且满足 j<k
- K2[i]        :: Int，等于 length(tri_pairs[i])，即 K_i = Σ_{j<k} B_{ijk}
- n_edges, n_triangles_total（统计信息）
"""
function build_triangles_from_ER(N::Int, p::Float64, seed::Int)
    rng = MersenneTwister(seed)

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
                jj, kk = (j < k) ? (j, k) : (k, j)
                if Aij[jj, kk]             # 三角形闭包
                    push!(pairs, (jj, kk))
                end
            end
        end
        tri_pairs[i] = pairs
        K2[i] = length(pairs)
    end

    n_tri_total = round(Int, sum(K2) / 3)  # 每个三角形被 i/j/k 计数三次
    return tri_pairs, K2, n_edges, n_tri_total
end

const TRI_DATA = build_triangles_from_ER(N, p_edge, NET_SEED)
const TRI_PAIRS = TRI_DATA[1]
const K2 = TRI_DATA[2]

# -----------------------------
# 参数封装
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
# 漂移项：二体 + 三体耦合 + 外驱动 + 噪声
# -----------------------------
function drift!(du, u, p::Params, t)
    N  = p.N
    α  = p.alpha
    σc = p.sigma
    τ  = p.tau

    x = @view u[1:N]
    ext = A * sin(Ω * t)

    # mean-field 二体项需要 sum(x)
    sumx = 0.0
    @inbounds for i in 1:N
        sumx += x[i]
    end

    η = (τ > 0) ? @view(u[N+1:2N]) : nothing

    @inbounds for i in 1:N
        xi = x[i]

        # 三体项：α * σ / K_i * Σ_{j<k} B_{ijk} tanh(x_j + x_k - 2x_i)
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

        # 二体项：全连接扩散耦合
        pair = ((1.0 - α) * σc / N) * (sumx - N * xi)

        noise_drive = (τ > 0) ? η[i] : 0.0
        du[i] = a * xi - b * xi^3 + ext + tri + pair + noise_drive
    end

    if τ > 0
        @inbounds for i in 1:N
            du[N + i] = -η[i] / τ
        end
    end

    return nothing
end

# -----------------------------
# 扩散项（对角噪声）
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
            dg[N + i, N + i] = sqrt(2.0 * D) / τ
        end
    end
    return nothing
end

# -----------------------------
# 单条路径模拟，返回节点均值时间序列
# -----------------------------
function simulate(seed, D, τ, σ, α)
    Random.seed!(seed)

    u0 = τ == 0 ? rand(Uniform(-1, 1), N) : vcat(rand(Uniform(-1, 1), N), zeros(N))
    noise_rate = zeros(length(u0), length(u0))

    p = Params(D, τ, σ, α, N, TRI_PAIRS, K2)
    prob = SDEProblem(drift!, diffusion!, u0, (0.0, T), p; noise_rate_prototype=noise_rate)

    sv = SavedValues(Float64, Float64)
    save_cb = SavingCallback((u, t, _) -> mean(@view u[1:N]), sv; saveat = save_dt)
    solve(prob, EM(); dt = dt, adaptive = false, callback = save_cb)
    sv.saveval
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 6 || (println("Usage:julia simulate_single_path.jl <seed> <alpha> <tau> <D> <path_idx> <outdir>"); exit(1))
    seed    = parse(Int, ARGS[1])
    alpha   = parse(Float64, ARGS[2])
    tau     = parse(Float64, ARGS[3])
    D       = parse(Float64, ARGS[4])
    path_idx= parse(Int, ARGS[5])
    outdir  = ARGS[6]

    result = simulate(seed + path_idx * 1000, D, tau, σ, alpha)
    mkpath(outdir)
    fname = joinpath(outdir, "alpha$(alpha)_tau$(tau)_D$(D)_path$(path_idx).mat")
    matwrite(fname, Dict("result" => result, "times" => collect(0:T)))
    @info "完成: $fname"
end
