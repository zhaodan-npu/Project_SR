#!/bin/bash
#SBATCH --job-name=FPTScanA1
#SBATCH --output=logs/fptA1_%A_%a.out
#SBATCH --error=logs/fptA1_%A_%a.err
#SBATCH --partition=standard
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# alpha(5) * tau(3) * D(51) = 765 tasks
#SBATCH --array=0-764%12

set -euo pipefail

mkdir -p logs

module purge
module load julia/1.12.2

export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${SLURM_SUBMIT_DIR}"

# fail-fast：缺包就立刻退出（你已经装好包后，这行几乎不耗时）
julia --project=. -e 'using DifferentialEquations, DiffEqCallbacks'

# -----------------------------
# 扫描参数
# -----------------------------
alpha_vals=(0.00 0.25 0.50 0.75 1.00)
tau_vals=(0.00 0.10 0.20)

# D = 0:0.02:1.00 -> 51 points
D_vals=()
for i in $(seq 0 50); do
  D_vals+=( "$(awk -v ii="$i" 'BEGIN{printf "%.2f", ii/50.0}')" )
done

NA=${#alpha_vals[@]}
NT=${#tau_vals[@]}
ND=${#D_vals[@]}

TOTAL=$((NA * NT * ND))
if [ "${TOTAL}" -ne 765 ]; then
  echo "[ERROR] TOTAL=${TOTAL} (expected 765). Please check grids."
  exit 1
fi

TASK=${SLURM_ARRAY_TASK_ID}

alpha_idx=$(( TASK / (NT*ND) ))
rem=$(( TASK % (NT*ND) ))
tau_idx=$(( rem / ND ))
d_idx=$(( rem % ND ))

alpha=${alpha_vals[$alpha_idx]}
tau=${tau_vals[$tau_idx]}
D=${D_vals[$d_idx]}

# -----------------------------
# Monte-Carlo 与输出
# -----------------------------
NPATHS=1000
SEEDBASE=20250101

OUTDIR="results_fpt_A1"
mkdir -p "${OUTDIR}"

echo "TASK=${TASK} alpha=${alpha} tau=${tau} D=${D} npaths=${NPATHS} seedbase=${SEEDBASE}"
echo "OUTDIR=${OUTDIR}"

# -----------------------------
# 运行（最终版 FPT 脚本）
# -----------------------------
julia --project=. scan_fpt_A1.jl \
  "${alpha}" "${tau}" "${D}" "${NPATHS}" "${SEEDBASE}" "${OUTDIR}"
