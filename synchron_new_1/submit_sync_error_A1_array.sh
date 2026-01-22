#!/bin/bash
#SBATCH --job-name=SyncErrA1
#SBATCH --output=logs/syncerrA1_%A_%a.out
#SBATCH --error=logs/syncerrA1_%A_%a.err
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# alpha(4) * tau(3) * D(101) = 1212 tasks
#SBATCH --array=0-1211%12

set -euo pipefail

mkdir -p logs
module purge
module load julia/1.12.2

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# 固定本工程 depot（与登录节点安装时一致）
export JULIA_DEPOT_PATH=$HOME/.julia_synchron_new_1
mkdir -p "$JULIA_DEPOT_PATH"

# 依赖检查：缺包就立刻退出，避免排队后才失败
julia --project=. -e 'using DifferentialEquations, DiffEqCallbacks'


# -----------------------------
# 扫描参数
# -----------------------------
alpha_vals=(0.0 0.25 0.5 0.75)
tau_vals=(0.0 0.1 0.2)

# D = 0:0.01:1.00 -> 101 个点（用格式化避免浮点串过长/误差）
D_vals=()
for i in $(seq 0 100); do
  D_vals+=( "$(awk -v ii="$i" 'BEGIN{printf "%.2f", ii/100.0}')" )
done

NA=${#alpha_vals[@]}
NT=${#tau_vals[@]}
ND=${#D_vals[@]}

TASK=${SLURM_ARRAY_TASK_ID}

alpha_idx=$(( TASK / (NT*ND) ))
rem=$(( TASK % (NT*ND) ))
tau_idx=$(( rem / ND ))
d_idx=$(( rem % ND ))

alpha=${alpha_vals[$alpha_idx]}
tau=${tau_vals[$tau_idx]}
D=${D_vals[$d_idx]}

NPATHS=20
SEEDBASE=20250101

OUTDIR="results_syncerr_A1"
mkdir -p "${OUTDIR}"

echo "TASK=${TASK} alpha=${alpha} tau=${tau} D=${D} npaths=${NPATHS}"

julia --project=. scan_sync_error_A1.jl \
  "${alpha}" "${tau}" "${D}" "${NPATHS}" "${SEEDBASE}" "${OUTDIR}"
