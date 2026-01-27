#!/bin/bash
#SBATCH --job-name=colored_sim_tau_D
#SBATCH --account=ff4
#SBATCH --output=logs/colored_sim_tau_D_%A_%a.out
#SBATCH --error=logs/colored_sim_tau_D_%A_%a.err
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-20

module load julia
export JULIA_DEPOT_PATH="$HOME/.julia"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd "$SLURM_SUBMIT_DIR"

OUTPUT_DIR="/p/tmp/junyouzh"  
mkdir -p "$OUTPUT_DIR" logs

SEED=$SLURM_ARRAY_TASK_ID
NUM_PATHS=1  # 每个作业对应一条路径，共20个作业共20条路径

echo "[${SEED}] 开始: $(date)"
julia --project=. simulate_task_combined.jl "$SEED" "$OUTPUT_DIR" "$NUM_PATHS"
echo "[${SEED}] 完成: $(date)"
