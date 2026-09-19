#!/bin/bash
#SBATCH --job-name=hbprobe2
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/jwj/tmp/claude-3158161/-scratch-jwj-research-GameWorldModel/d05a283e-9584-4f1d-933c-4ba93a6aaaa2/scratchpad/probe/%x_%j.out
P=/scratch/jwj/tmp/claude-3158161/-scratch-jwj-research-GameWorldModel/d05a283e-9584-4f1d-933c-4ba93a6aaaa2/scratchpad/probe
OUT=$P/result_${SLURM_JOB_ID}; mkdir -p "$OUT"
echo "=== NODE ==="; hostname; date; echo "SLURM_TMPDIR=$SLURM_TMPDIR"; nproc
GPU=0; if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "^GPU"; then GPU=1; nvidia-smi -L; fi; echo "GPU=$GPU"
module --force purge; module load StdEnv/2023 nodejs/20.16.0 2>&1; node -v
WORK=$SLURM_TMPDIR/pwtest; mkdir -p "$WORK"; cd "$WORK"
t0=$(date +%s); tar -xf $P/pwstage.tar; echo "untar took $(( $(date +%s) - t0 ))s"
export PLAYWRIGHT_BROWSERS_PATH=$WORK/pw-browsers
CHROME=$(find "$PLAYWRIGHT_BROWSERS_PATH" -maxdepth 3 -type f \( -name chrome -o -name headless_shell \) | head -3); echo "chrome binaries: $CHROME"
for c in $CHROME; do echo "--- ldd $c ---"; ldd "$c" | grep "not found" || echo "no missing libs"; done
cp $P/probe_render.mjs "$WORK/"
echo "=== HEADLESS RENDER TEST (GPU=$GPU) ==="
timeout 900 node probe_render.mjs "$WORK" "$OUT" "$GPU" 2>&1 | tail -120
echo "=== DONE ==="; date
