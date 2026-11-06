#!/bin/bash
# Creates one folder per gouge angle, each with a copy of ShearApparatus_v9.jl running that single angle
# and a SLURM submission script. Run from the folder that contains ShearApparatus_v9.jl:
#   ./setup_runs.sh            # create the folders only
#   ./setup_runs.sh --submit   # create the folders and sbatch every job
#   ./setup_runs.sh 85 --submit # only the listed angle(s)
set -euo pipefail

ANGLES=(15 20 30 40 50 60 70 80 85)
SUBMIT=
args=()
for x in "$@"; do [[ $x == --submit ]] && SUBMIT=1 || args+=("$x"); done
(( ${#args[@]} )) && ANGLES=("${args[@]}")
SCRIPT=ShearApparatus_v9.jl
JULIA=julia                                  # or the full path to the julia binary
PROJECT=/work/amdjulia/duretz/StagFDTools.jl # Julia environment with StagFDTools (folder with Project.toml)

[[ -f $SCRIPT ]] || { echo "$SCRIPT not found in $(pwd)"; exit 1; }

for a in "${ANGLES[@]}"; do
    dir=$(printf "Tet%02d" "$a")
    mkdir -p "$dir"

    # Angle list of the run block -> single angle (the commented '# θ = ...' line is left untouched)
    sed -E "s/^([[:space:]]*)θ[[:space:]]*=[[:space:]]*\[.*$/\1θ  = [${a}.0]/" "$SCRIPT" > "$dir/$SCRIPT"
    n=$(grep -cE "^[[:space:]]*θ  = \[${a}\.0\]$" "$dir/$SCRIPT" || true)
    [[ $n == 1 ]] || { echo "$dir: expected 1 angle line, found $n"; exit 1; }

    cat > "$dir/submit.sh" <<EOF
#!/bin/bash
#SBATCH --account=agmisc
#SBATCH --job-name=SA_Tet$(printf "%02d" "$a")
#SBATCH --partition=fuchs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1024
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --mail-type=FAIL
#SBATCH --output=slurm-%j.out

srun hostname

export JULIA_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1

cd "\$SLURM_SUBMIT_DIR"
$JULIA --project=$PROJECT -t \$SLURM_CPUS_PER_TASK $SCRIPT
EOF
    chmod +x "$dir/submit.sh"
    echo "created $dir"

    if [[ -n $SUBMIT ]]; then
        (cd "$dir" && sbatch submit.sh)
    fi
done
