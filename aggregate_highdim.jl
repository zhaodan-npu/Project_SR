#!/usr/bin/env julia
using MAT

OUTPUT_DIR="/p/tmp/junyouzh"
alpha_vals=[0.0,0.25,0.5,0.75]
tau_vals=[0.0,0.1,0.2]
D_vals=collect(0.0:0.02:1.0) # 共51个
NUM_PATHS=20
time_points=2001 # 0到2000共2001个点

# 初始化高维数组(alpha,tau,D,path,time)
combined=zeros(length(alpha_vals),length(tau_vals),length(D_vals),NUM_PATHS,time_points)

for (ai,alpha) in enumerate(alpha_vals), 
    (ti,tau) in enumerate(tau_vals), 
    (di,D) in enumerate(D_vals), 
    pi in 0:NUM_PATHS-1

    fname=joinpath(OUTPUT_DIR,"alpha$(alpha)_tau$(tau)_D$(D)_path$(pi).mat")
    if isfile(fname)
        combined[ai,ti,di,pi+1,:]=matread(fname)["result"]
    else
        @warn "文件不存在: $fname"
    end
end

# 存储合并后的数据
matwrite(joinpath(OUTPUT_DIR,"combined_highdim.mat"),
    Dict("alpha_vals"=>alpha_vals,
         "tau_vals"=>tau_vals,
         "D_vals"=>D_vals,
         "times"=>collect(0:2000),
         "combined"=>combined))

println("数据合并完成，存储为 combined_highdim.mat，数组尺寸为：", size(combined))
