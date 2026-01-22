#!/usr/bin/env julia
using Random
using Statistics
using DifferentialEquations
using MAT

# ---------------------------------
# 常数参数（与 synchron_new_1 对齐）
# ---------------------------------
const a = 1.0
const b = 1.0
const A = 0.2
const Ω = 0.1

const N  = 50
const σ  = 0.1

const dt      = 0.01
const T       = 2000.0
const save_dt = 1.0
const t_trans = 500.0          # 计算同步误差的稳态起点
const T_FPT   = 200.0          # 首穿统计窗口（与 mfpt_new_1 一致）

# 首穿时间定义（与 mfpt_new_1 一致）
const X_INIT = -1.0            # 初值：全体在左井
const X_THR  = 0.0             # 首次越过阈值

# 三体拓扑：ER 图 + 三角形闭包（simplicial complex）
const p_edge   = 0.30
const NET_SEED = 20250101

# ---------------------------------
# 三角形张量生成（与 synchron_new_1 相同）
# ---------------------------------
"""
build_triangles_from_ER(N, p_edge, seed)
返回：
- tri_pairs[i] :: Vector{NTuple{2,Int}}，存 (j,k) 且 j<k, 满足 B_{ijk}=1
- K2[i]        :: Int，K_i = Σ_{j<k} B_{ijk}
- n_edges, n_triangles_total 统计
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
                if Aij[jj, kk]
                    push!(pairs, (jj, kk))
                end
            end
        end
        tri_pairs[i] = pairs
        K2[i] = length(pairs)
    end

    n_tri_total = round(Int, sum(K2) / 3)
    return tri_pairs, K2, n_edges, n_tri_total
end

const TRI_DATA = build_triangles_from_ER(N, p_edge, NET_SEED)
const TRI_PAIRS = TRI_DATA[1]
const K2 = TRI_DATA[2]

# ---------------------------------
# 参数封装
# ---------------------------------
mutable struct Params
    D::Float64
    tau::Float64
    sigma::Float64
    alpha::Float64
    N::Int
    tri_pairs::Vector{Vector{NTuple{2,Int}}}
    K2::Vector{Int}
end

# ---------------------------------
# 漂移项：二体 + 三体 + 外驱动 + 色噪声
# ---------------------------------
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

# ---------------------------------
# 扩散项（对角噪声）
# ---------------------------------
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

# ---------------------------------
# 单条路径：返回均值序列、同步误差、Ebar、首穿时间
# ---------------------------------
function simulate_path(seed::Int, α::Float64, τ::Float64, D::Float64; x_init::Float64=X_INIT, x_thr::Float64=X_THR)
    Random.seed!(seed)

    u0 = (τ == 0.0) ? fill(x_init, N) : vcat(fill(x_init, N), zeros(N))
    noise_rate = zeros(length(u0), length(u0))

    p = Params(D, τ, σ, α, N, TRI_PAIRS, K2)
    prob = SDEProblem(drift!, diffusion!, u0, (0.0, T), p; noise_rate_prototype=noise_rate)

    integ = init(prob, EM(); dt=dt, adaptive=false, save_everystep=false)

    t_vec = Float64[0.0]
    mean_vec = Float64[mean(view(u0, 1:N))]
    sync_vec = Float64[0.0]   # 初始所有节点相等，方差为 0

    n_steps = Int(round(T / dt))
    save_every = max(1, Int(round(save_dt / dt)))

    t_hit = NaN

    for k in 1:n_steps
        step!(integ)
        x = @view integ.u[1:N]

        sumx = 0.0
        @inbounds for i in 1:N
            sumx += x[i]
        end
        μ = sumx / N

        if isnan(t_hit) && μ >= x_thr
            t_hit = integ.t
        end

        if (k % save_every == 0) || (k == n_steps)
            sse = 0.0
            @inbounds for i in 1:N
                d = x[i] - μ
                sse += d * d
            end
            push!(t_vec, integ.t)
            push!(mean_vec, μ)
            push!(sync_vec, sse / N)
        end
    end

    hitflag = (!isnan(t_hit) && (t_hit <= T_FPT))
    t_fpt = hitflag ? t_hit : NaN
    t_fpt_cens = hitflag ? t_hit : T_FPT

    # Ebar：稳态区间（t >= t_trans）同步误差平均
    idx = findall(t -> t >= t_trans, t_vec)
    Ebar = !isempty(idx) ? mean(@view sync_vec[idx]) : NaN

    return (times = t_vec,
            mean = mean_vec,
            sync = sync_vec,
            Ebar = Ebar,
            hit = hitflag ? 1 : 0,
            t_fpt = t_fpt,
            t_fpt_censored = t_fpt_cens)
end

# ---------------------------------
# 工具：解析逗号分隔的参数列表
# ---------------------------------
function parse_list(str::AbstractString, ::Type{T}) where {T<:Number}
    parts = split(str, ",")
    vals = T[]
    for s in parts
        s = strip(s)
        isempty(s) && continue
        push!(vals, parse(T, s))
    end
    return vals
end

# ---------------------------------
# main
# ---------------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) != 6
        println("Usage:\n  julia simulate_single_path.jl <seedbase> <alpha[,..]> <tau[,..]> <D[,..]> <path_idx[,..]> <outdir>")
        println("Example (single):  julia simulate_single_path.jl 1 0.5 0.1 0.2 0 /tmp/out")
        println("Example (batch):   julia simulate_single_path.jl 1 0.25,0.5 0.0,0.1 0.1,0.2 0,1 /tmp/out")
        exit(1)
    end

    seedbase = parse(Int, ARGS[1])
    alphas   = parse_list(ARGS[2], Float64)
    taus     = parse_list(ARGS[3], Float64)
    Ds       = parse_list(ARGS[4], Float64)
    paths    = parse_list(ARGS[5], Int)
    outdir   = ARGS[6]

    any(length.([alphas, taus, Ds, paths]) .== 0) && error("alpha/tau/D/path 列表不能为空")

    mkpath(outdir)

    for α in alphas, τ in taus, D in Ds, path_idx in paths
        αtag = round(Int, α * 1000)
        τtag = round(Int, τ * 1000)
        Dtag = round(Int, D * 1000)
        seed = seedbase + αtag * 1_000_000 + τtag * 1_000 + Dtag * 10 + path_idx

        res = simulate_path(seed, α, τ, D)

        fname = joinpath(outdir, "alpha$(α)_tau$(τ)_D$(D)_path$(path_idx).mat")
        matwrite(fname, Dict(
            "result" => res.mean,   # 兼容旧版字段名
            "mean" => res.mean,
            "sync" => res.sync,
            "times" => res.times,
            "Ebar" => res.Ebar,
            "hit" => res.hit,
            "t_fpt" => res.t_fpt,
            "t_fpt_censored" => res.t_fpt_censored,
            "params" => Dict(
                "alpha"=>α, "tau"=>τ, "D"=>D, "path"=>path_idx, "seed"=>seed,
                "sigma"=>σ, "a"=>a, "b"=>b, "A"=>A, "Omega"=>Ω,
                "N"=>N, "dt"=>dt, "save_dt"=>save_dt, "T"=>T, "t_trans"=>t_trans,
                "T_FPT"=>T_FPT,
                "x_init"=>X_INIT, "x_thr"=>X_THR,
                "p_edge"=>p_edge, "net_seed"=>NET_SEED
            )
        ))
        @info "done" alpha=α tau=τ D=D path=path_idx seed=seed hit=res.hit t_fpt=res.t_fpt Ebar=res.Ebar file=fname
    end
end
