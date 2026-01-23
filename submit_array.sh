#!/bin/bash
#SBATCH --job-name=sim_large_array
#SBATCH --account=ff4
#SBATCH --qos=medium
#SBATCH --output=logs/sim_%A_%a.out
#SBATCH --error=logs/sim_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

set -euo pipefail

# ---- Julia 配置 ----
# 如果集群提供 julia 模块，保持 module load；否则将 JULIA_CMD 改为实际安装路径
module purge
module load julia
JULIA_CMD=${JULIA_CMD:-$(which julia)}
JULIA_PROJECT=${JULIA_PROJECT:-.}   # 如有自定义环境，改成其路径

export JULIA_DEPOT_PATH="$HOME/.julia"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

cd "$SLURM_SUBMIT_DIR"

SEED=1
alpha_vals=(0.0 0.25 0.5 0.75)
tau_vals=(0.0 0.1 0.2)
D_vals=($(seq 0.0 0.02 1.0))
NUM_PATHS=20
# 将输出写到提交作业的当前目录，避免写到其他路径
OUTPUT_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$OUTPUT_DIR" logs

# 用 OFFSET 支持分批提交（默认 0）
OFFSET=${OFFSET:-0}

i=$(( SLURM_ARRAY_TASK_ID + OFFSET ))

nD=${#D_vals[@]}
ntau=${#tau_vals[@]}
np=$NUM_PATHS
block_tauD=$((ntau * nD * np))

alpha_idx=$(( i / block_tauD ))
rem1=$(( i % block_tauD ))
tau_idx=$(( rem1 / (nD * np) ))
rem2=$(( rem1 % (nD * np) ))
D_idx=$(( rem2 / np ))
path_idx=$(( rem2 % np ))

alpha=${alpha_vals[$alpha_idx]}
tau=${tau_vals[$tau_idx]}
D=${D_vals[$D_idx]}

echo "alpha=$alpha tau=$tau D=$D path=$path_idx"
"$JULIA_CMD" --project="$JULIA_PROJECT" simulate_single_path.jl \
  $SEED $alpha $tau $D $path_idx "$OUTPUT_DIR"
