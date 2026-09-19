#!/bin/bash
#SBATCH --job-name=gwm-tests
#SBATCH --account=aip-zhouyang
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/jwj/gwm/logs/%x_%j.out
source /scratch/jwj/research/GameWorldModel/scripts/setup_env.sh
cd /scratch/jwj/research/GameWorldModel && python -m pytest -q tests 2>&1 | tail -20
