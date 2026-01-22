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


module load julia
export JULIA_DEPOT_PATH="$HOME/.julia"
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd "$SLURM_SUBMIT_DIR"

SEED=1
alpha_vals=(0.0 0.25 0.5 0.75)
tau_vals=(0.0 0.1 0.2)
D_vals=($(seq 0.0 0.02 1.0))
NUM_PATHS=20
OUTPUT_DIR="/p/tmp/junyouzh"
mkdir -p "$OUTPUT_DIR" logs

OFFSET=${OFFSET:-0}  # 如果未设置OFFSET，默认为0

i=$(($SLURM_ARRAY_TASK_ID + $OFFSET))

nD=${#D_vals[@]}
ntau=${#tau_vals[@]}
np=$NUM_PATHS
block_tauD=$((ntau*nD*np))

alpha_idx=$(( i / block_tauD ))
rem1=$(( i % block_tauD ))
tau_idx=$(( rem1 / (nD*np) ))
rem2=$(( rem1 % (nD*np) ))
D_idx=$(( rem2 / np ))
path_idx=$(( rem2 % np ))

alpha=${alpha_vals[$alpha_idx]}
tau=${tau_vals[$tau_idx]}
D=${D_vals[$D_idx]}

echo "alpha=$alpha tau=$tau D=$D path=$path_idx"
julia --project=. simulate_single_path.jl $SEED $alpha $tau $D $path_idx $OUTPUT_DIR
