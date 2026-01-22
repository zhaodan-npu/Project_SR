#!/usr/bin/env julia
using Random, Distributions, DifferentialEquations, DiffEqCallbacks, MAT

const m, A, ω, N, dt, T = 1.0, 0.2, 0.1, 50, 0.01, 2000.0
const σ = 0.1
const sinωt = @. A*sin(ω*(0:dt:T-dt))
const inv2N2 = 1.0/(2N*N)

function coupling!(f,x,σ,α)
    σ_tri = α*σ*inv2N2
    σ_pair = (1-α)*σ/N
    for i in 1:N
        s_tri,s_pair = 0.0,0.0
        for j in 1:N, k in 1:N
            if i!=j && i!=k
                s_tri+=tanh(x[j]+x[k]-2x[i])
            end
        end
        for j in 1:N
            if i!=j
                s_pair+=(x[j]-x[i])
            end
        end
        f[i]=σ_tri*s_tri+σ_pair*s_pair
    end
end

function drift!(du,u,p,t)
    D,τ,σ,α=p; x=@view u[1:N]; ext=sinωt[Int(floor(t/dt+1e-9))+1]
    coupling!(du,x,σ,α)
    if τ==0
        @. du+=m*x-x^3+ext
    else
        η=@view u[N+1:end]
        for i=1:N
            du[i]+=m*x[i]-x[i]^3+ext+η[i]
            du[N+i]=-η[i]/τ
        end
    end
end

function diffusion!(dg,u,p,t)
    D,τ,_,_=p;dg.=0.0
    if τ==0
        for i=1:N;dg[i,i]=sqrt(2D);end
    else
        for i=1:N;dg[N+i,N+i]=sqrt(2)*sqrt(D)/τ;end
    end
end

function simulate(seed,D,τ,σ,α)
    Random.seed!(seed)
    u0=τ==0 ? rand(Uniform(-1,1),N) : vcat(rand(Uniform(-1,1),N),zeros(N))
    noise_rate=τ==0 ? zeros(N,N) : zeros(2N,2N)
    prob=SDEProblem(drift!,diffusion!,u0,(0.0,T),(D,τ,σ,α);noise_rate_prototype=noise_rate)
    sv=SavedValues(Float64,Float64)
    save_cb=SavingCallback((u,t,_)->mean(u[1:N]),sv;saveat=1.0)
    solve(prob,EM();dt=dt,adaptive=false,callback=save_cb)
    sv.saveval
end

if abspath(PROGRAM_FILE)==@__FILE__
    length(ARGS)==6||(println("Usage:julia simulate_single_path.jl <seed> <alpha> <tau> <D> <path_idx> <outdir>");exit(1))
    seed=parse(Int,ARGS[1])
    alpha=parse(Float64,ARGS[2])
    tau=parse(Float64,ARGS[3])
    D=parse(Float64,ARGS[4])
    path_idx=parse(Int,ARGS[5])
    outdir=ARGS[6]
    result=simulate(seed+path_idx*1000,D,tau,σ,alpha)
    mkpath(outdir)
    fname=joinpath(outdir,"alpha$(alpha)_tau$(tau)_D$(D)_path$(path_idx).mat")
    matwrite(fname,Dict("result"=>result,"times"=>collect(0:T)))
    @info "完成: $fname"
end
