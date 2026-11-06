#!/bin/bash
# Restart the TetXX runs from their most recent readable checkpoint. Run from the folder containing the TetXX folders
# (check_checkpoints.jl must be there too):
#   ./restart_runs.sh            # dry run: kill nothing, change nothing, print the restart step of each run
#   ./restart_runs.sh --submit   # cancel ALL my jobs, set restart=<step> and the time limit, resubmit
set -euo pipefail

PROJECT=/work/amdjulia/duretz/StagFDTools.jl
TIME=10:00:00
SCRIPT=ShearApparatus_v9.jl
SUBMIT=; [[ ${1:-} == --submit ]] && SUBMIT=1

if [[ -n $SUBMIT ]]; then
    echo "Cancelling all jobs of $USER"
    scancel -u "$USER" || true
    for _ in $(seq 60); do   # wait (max 5 min) until they are gone, so that no job is still writing a checkpoint
        [[ -z $(squeue -u "$USER" -h) ]] && break
        sleep 5
    done
    [[ -z $(squeue -u "$USER" -h) ]] || { echo "jobs still in the queue, aborting"; squeue -u "$USER"; exit 1; }
fi

# Most recent readable checkpoint of every run (unreadable ones are renamed *.corrupt, only with --submit)
plan=$(julia --project="$PROJECT" check_checkpoints.jl ${SUBMIT:+--rename} Tet*/)
echo "$plan" | awk 'BEGIN{printf "%-8s %-6s %6s %10s\n", "run", "status", "step", "time [s]"} {printf "%-8s %-6s %6s %10s\n", $1, $2, $3, $4}'

while read -r dir status step tsec; do
    case $status in
        done) echo "$dir: time_max reached, not resubmitted"; continue ;;
        none) echo "$dir: NO readable checkpoint, not resubmitted (check it by hand)"; continue ;;
    esac
    [[ -n $SUBMIT ]] || continue

    # restart=<step> in the main() call of the run block (the default in the function signature is not touched)
    sed -i -E "s/^([[:space:]]*)restart=[0-9]+,/\1restart=$step,/" "$dir/$SCRIPT"
    n=$(grep -cE "^[[:space:]]*restart=$step," "$dir/$SCRIPT" || true)
    [[ $n == 1 ]] || { echo "$dir: expected 1 restart line, found $n, not resubmitted"; continue; }

    sed -i -E "s/^#SBATCH --time=.*/#SBATCH --time=$TIME/" "$dir/submit.sh"
    sed -i -E "s|--project=[^ ]*|--project=$PROJECT|" "$dir/submit.sh"
    (cd "$dir" && sbatch submit.sh)
    echo "$dir: resubmitted, restart=$step"
done <<< "$plan"
