#!/bin/bash
#SBATCH --job-name=hbprobe
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/jwj/tmp/claude-3158161/-scratch-jwj-research-GameWorldModel/d05a283e-9584-4f1d-933c-4ba93a6aaaa2/scratchpad/probe/%x_%j.out

PROBE_DIR=/scratch/jwj/tmp/claude-3158161/-scratch-jwj-research-GameWorldModel/d05a283e-9584-4f1d-933c-4ba93a6aaaa2/scratchpad/probe
OUT=$PROBE_DIR/result_${SLURM_JOB_ID}
mkdir -p "$OUT"
echo "=== NODE INFO ==="; hostname; date; echo "SLURM_TMPDIR=$SLURM_TMPDIR"; echo "http_proxy=$http_proxy https_proxy=$https_proxy no_proxy=$no_proxy"
echo "XDG_CACHE_HOME=$XDG_CACHE_HOME"; nproc; free -g | head -2
GPU=0; if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then GPU=1; echo "=== GPU ==="; nvidia-smi -L; nvidia-smi --query-gpu=driver_version --format=csv,noheader; fi
echo "GPU=$GPU"

echo "=== MODULES ==="
module --force purge
module load StdEnv/2023 nodejs/20.16.0 2>&1
node -v; npm -v
module load apptainer 2>&1 && apptainer --version

echo "=== NETWORK (via proxy env) ==="
for u in https://registry.npmjs.org/ https://cdn.playwright.dev/ https://playwright.download.prss.microsoft.com/ https://github.com/ https://huggingface.co/ https://api.anthropic.com/ https://api.openai.com/ https://poly.pizza/ https://sketchfab.com/ https://inference.vulcan.alliancecan.ca/; do
  code=$(curl -sS -o /dev/null -m 15 -w "%{http_code}" "$u" 2>&1 | tail -c 40); echo "proxy   $code  $u"
done
echo "--- direct (no proxy) ---"
for u in https://registry.npmjs.org/ https://api.anthropic.com/; do
  code=$(env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY curl -sS -o /dev/null -m 10 -w "%{http_code}" "$u" 2>&1 | tail -c 60); echo "direct  $code  $u"
done

echo "=== CHROMIUM SHARED LIBS (system ldconfig) ==="
for lib in libnss3.so libnssutil3.so libsmime3.so libatk-1.0.so.0 libatk-bridge-2.0.so.0 libcups.so.2 libdrm.so.2 libxkbcommon.so.0 libXcomposite.so.1 libXdamage.so.1 libXrandr.so.2 libgbm.so.1 libpango-1.0.so.0 libasound.so.2 libxshmfence.so.1 libatspi.so.0 libX11.so.6 libEGL.so.1 libGL.so.1 libvulkan.so.1 libGLESv2.so.2; do
  if ldconfig -p 2>/dev/null | grep -q "$lib"; then echo "found   $lib"; else echo "MISSING $lib"; fi
done
ls /usr/lib/x86_64-linux-gnu/libEGL_nvidia* /usr/lib/x86_64-linux-gnu/libnvidia-egl* /usr/share/glvnd/egl_vendor.d/ /usr/share/vulkan/icd.d/ 2>&1 | head -20

echo "=== PLAYWRIGHT + THREE.JS INSTALL (in SLURM_TMPDIR) ==="
export npm_config_cache=$SLURM_TMPDIR/npm-cache
export PLAYWRIGHT_BROWSERS_PATH=$SLURM_TMPDIR/pw-browsers
WORK=$SLURM_TMPDIR/pwtest; mkdir -p "$WORK"; cd "$WORK"
npm init -y >/dev/null 2>&1
t0=$(date +%s); npm install --no-audit --no-fund playwright three 2>&1 | tail -3; echo "npm install took $(( $(date +%s) - t0 ))s"
node -e "console.log('playwright', require('playwright/package.json').version, 'three', require('three/package.json').version)"
t0=$(date +%s); npx playwright install chromium 2>&1 | tail -5; echo "browser download took $(( $(date +%s) - t0 ))s"
du -sh "$PLAYWRIGHT_BROWSERS_PATH" 2>/dev/null
CHROME=$(find "$PLAYWRIGHT_BROWSERS_PATH" -maxdepth 3 -type f -name chrome | head -1); echo "chrome binary: $CHROME"
if [ -n "$CHROME" ]; then echo "--- ldd missing ---"; ldd "$CHROME" | grep "not found" || echo "no missing libs for chrome binary"; fi

echo "=== HEADLESS RENDER TEST ==="
cp "$PROBE_DIR/probe_render.mjs" "$WORK/"
timeout 600 node probe_render.mjs "$WORK" "$OUT" "$GPU" 2>&1 | tail -80

echo "=== APPTAINER PULL TEST (small image via proxy) ==="
export APPTAINER_CACHEDIR=$SLURM_TMPDIR/apptainer-cache; export APPTAINER_TMPDIR=$SLURM_TMPDIR/apptainer-tmp; mkdir -p $APPTAINER_CACHEDIR $APPTAINER_TMPDIR
timeout 180 apptainer exec docker://alpine:3.20 echo "apptainer docker pull+exec OK" 2>&1 | tail -3

echo "=== DONE ==="; date
