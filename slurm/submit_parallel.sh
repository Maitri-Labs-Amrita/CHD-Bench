#!/bin/bash
#SBATCH --job-name=chd-batch-par
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --signal=B:TERM@120
#SBATCH --time=02:00:00
#SBATCH --output=slurm/batches/slurm-par-%j.out
set -euo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
    [ $# -ge 1 ] || { echo "usage: $0 <config.json> [--dry-run-scripted]" >&2; exit 2; }
    self="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"; cfg="$(realpath "$1")"
    cd "$(dirname "$self")/.."
    mkdir -p slurm/batches
    python3 -c 'import json,sys; c=json.load(open(sys.argv[1])); dry=sys.argv[2]=="dry"; jobs=c["jobs"]
n=len(jobs); g=max(1, min(int(c.get("slurm_jobs", 1)), n)); first=0
for i in range(g):
    size=n // g + (i < n % g); part=jobs[first:first + size]
    m=30 if dry else min(720, max(len(j["cases"]) for j in part) * (c.get("episode_timeout_s", 5400) + 120) // 60 + 10)
    print(first, size, f"{m // 60:02d}:{m % 60:02d}:00"); first += size' "$cfg" \
        "$([[ " ${*:2} " == *" --dry-run-scripted "* ]] && echo dry || echo real)" |
    while read -r first size t; do
        sbatch --ntasks="$size" --time="$t" --export=ALL,CHD_BATCH_FIRST="$first" "$self" "$cfg" "${@:2}"
    done
    exit 0
fi
cd "$SLURM_SUBMIT_DIR"
if [[ " $* " == *" --dry-run-scripted "* ]]; then unset ANTHROPIC_API_KEY OPENROUTER_API_KEY; fi
export PATH="$HOME/.local/bin:$PATH"
export CUDA_VISIBLE_DEVICES=""
export LP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS="$SLURM_CPUS_PER_TASK"
export UV_NO_SYNC=1
pids=()
trap 'kill -TERM "${pids[@]}" 2>/dev/null' TERM
first="${CHD_BATCH_FIRST:-0}"
for k in $(seq "$first" $((first + SLURM_NTASKS - 1))); do
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exact --output="slurm/batches/slurm-par-${SLURM_JOB_ID}_${k}.out" \
        uv run --frozen python slurm/run_batch.py --config "$1" --job "$k" "${@:2}" &
    pids+=($!)
done
rc=0
for p in "${pids[@]}"; do
    while true; do wait "$p" && break; kill -0 "$p" 2>/dev/null || { rc=1; break; }; done
done
exit $rc
