#!/usr/bin/env julia
using Random, Distributions, DifferentialEquations, DiffEqCallbacks, MAT

# 固定参数定义
const m, A, ω, N, dt, T, σ, α = 1.0, 0.2, 0.1, 50, 0.01, 2000.0, 0.1, 1.0
const tgrid = collect(0:dt:T-dt)

# 变化参数定义
const tau_vals = collect(0.0:0.1:1.0)
const D_vals = collect(0.0:0.04:2.0)

const sinωt = @. A*sin(ω*tgrid)
const inv2N2 = 1.0/(2N*N)

# 耦合函数
function coupling!(f,x,σ,α)
    σ_tri = α*σ*inv2N2
    for i in 1:N
        s_tri = 0.0
        for j in 1:N,k in 1:N
            if i!=j && i!=k
                s_tri += tanh(x[j]+x[k]-2x[i])
            end
        end
        f[i]=σ_tri*s_tri
    end
end

# 漂移函数
function drift!(du,u,p,t)
    D,τ,σ,α=p
    x=@view u[1:N]
    η=@view u[N+1:end]

    coupling!(du,x,σ,α)
    ext=sinωt[Int(floor(t/dt+1e-9))+1]

    if τ==0
        @. du[1:N] += m*x - x^3 + ext
    else
        @. du[1:N] += m*x - x^3 + ext + η
        @. du[N+1:end] = -η/τ
    end
end

# 扩散函数
function diffusion!(dg,u,p,t)
    D,τ,_,_=p
    dg .= 0.0
    if τ==0
        for i in 1:N
            dg[i,i]=sqrt(2D)
        end
    else
        for i in 1:N
            dg[N+i,N+i]=sqrt(2D/τ)
        end
    end
end

# 单路径仿真函数
function simulate_one(seed,D,τ,σ,α)
    Random.seed!(seed)
    u0 = τ==0 ? rand(Uniform(-0.75,0.75),N) : vcat(rand(Uniform(-0.75,0.75),N),zeros(N))
    noise_rate=τ==0 ? zeros(N,N) : zeros(2N,2N)
    prob=SDEProblem(drift!,diffusion!,u0,(0.0,T),(D,τ,σ,α);noise_rate_prototype=noise_rate)
    sv=SavedValues(Float64,Float64)
    save_cb=SavingCallback((u,t,_)->mean(u[1:N]),sv;saveat=1.0)
    solve(prob,EM();dt=dt,adaptive=false,callback=save_cb)
    return sv.saveval
end

# 完整参数仿真函数（修正保存路径问题）
function simulate_all(seed,outdir,num_paths)
    num_tau,num_D,num_t=length(tau_vals),length(D_vals),Int(T)+1
    results=zeros(num_tau,num_D,num_paths,num_t)

    for (τ_idx,τ) in enumerate(tau_vals)
        @info "tau=$τ"
        for (D_idx,D) in enumerate(D_vals)
            for path_idx in 1:num_paths
                myseed=seed+path_idx*1000+D_idx*10000+τ_idx*100000
                results[τ_idx,D_idx,path_idx,:].=simulate_one(myseed,D,τ,σ,α)
            end
        end
    end

    mkpath(outdir)
    fname=joinpath(outdir,"fig7_seed$(seed).mat")  # ←←← 修正的地方
    matwrite(fname,Dict(
        "tau_vals"=>tau_vals,"D_vals"=>D_vals,
        "times"=>collect(0:T),"results"=>results
    ))
    @info "存储完成: $fname"
end

# Main调用
if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==3||(println("Usage:julia script.jl<seed><outdir><num_paths>");exit(1))
    seed,outdir,num_paths=parse.(Int,ARGS[1]),ARGS[2],parse(Int,ARGS[3])
    simulate_all(seed,outdir,num_paths)
    @info "仿真完成;数据存于:$outdir"
end
