#!/bin/bash
#SBATCH -J ABPO
#SBATCH -A loni_pdrug -p workq -N 3
#SBATCH --ntasks-per-node=4 --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o /work/jcarde7/polyz2/3_simulations/generation_01/out_%j.txt -e /work/jcarde7/polyz2/3_simulations/generation_01/err_%j.txt
NAMD=/work/jcarde7/polyz2/NAMD_2.14_Linux-x86_64-multicore/namd2; BASE=/work/jcarde7/polyz2/3_simulations/generation_01; SIMS=(child_01 child_02 child_03 child_04 child_05 child_06 child_07 child_08 child_09 child_10 child_11 child_12)
for s in "${SIMS[@]}"; do
  srun -N1 --ntasks=1 --cpus-per-task=16 --exclusive "$NAMD" +p16 "$BASE/$s/config.namd" \
       >"$BASE/$s/output.txt" 2>&1 &
done
wait
