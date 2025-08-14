#!/usr/bin/env python3
"""
ABPO – Adaptive Batch Parameter Optimisation
--------------------------------------------
Optimises bead-type choices (C-bead, N-bead, Q+ and Q–; each in {1..6})
to match a target radius of gyration using 12 parallel NAMD simulations
per generation.

Replace your previous CMA-ES `main.py` with this file; everything else
(directory creation, SLURM submission, Rg calculation) is reused.
"""
from __future__ import annotations

import csv
import itertools
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

import MDAnalysis as mda
import numpy as np
import paramiko

# ---------------------------------------------------------------------- #
# ----------------------------  CONSTANTS  ----------------------------- #
# ---------------------------------------------------------------------- #

BASE = Path("/work/jcarde7/polyzwitterion")              # HPC workdir
POP  = 12                                               # batch size
TARGET_RG  = 36.0
LOWER_RG   = 35.0
UPPER_RG   = 37.0
GEN_MAX    = 100                                        # safety cap
TOTAL_CAP  = 1200                                       # absolute run budget

REPO = Path(__file__).resolve().parent.parent
NAMD = BASE / "NAMD_2.14_Linux-x86_64-multicore/namd2"

# Discrete levels
C_LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6"]
N_LEVELS = ["N1", "N2", "N3", "N4", "N5", "N6"]
Q_LEVELS = ["Q1", "Q2", "Q3", "Q4", "Q5", "Q6"]

# Probability damping / batch recipe
DAMP          = 0.2          # 0 < DAMP <= 1
TOP_FRAC      = 0.25
EXPLOIT_K     = 8            # of 12 slots
RESEED_CHANCE = 0.05         # chance to resample any prob dist element to 1/6

# ---------------------------------------------------------------------- #
# -------------------------  SSH / CLUSTER UTILS  ---------------------- #
# ---------------------------------------------------------------------- #

_SSH_CLIENT: paramiko.SSHClient | None = None


def _get_ssh() -> paramiko.SSHClient:
    global _SSH_CLIENT
    if _SSH_CLIENT is None:
        c = paramiko.SSHClient()
        c.load_system_host_keys()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect("qbd.loni.org", username="jcarde7")
        _SSH_CLIENT = c
    return _SSH_CLIENT


def _ssh_exec(cmd: str) -> str:
    _, stdout, _ = _get_ssh().exec_command(cmd)
    return stdout.read().decode().strip()


# ---------------------------------------------------------------------- #
# ---------------------------  FILE HELPERS  --------------------------- #
# ---------------------------------------------------------------------- #

def create_generation(gen: int) -> Path:
    gdir = REPO / "3_simulations" / f"generation_{gen:02d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, POP + 1):
        cdir = gdir / f"child_{i:02d}"
        cdir.mkdir(exist_ok=True)

        # copy only the files that are truly local to the run
        shutil.copy(REPO / "0_parameters" / "config.namd", cdir)
        shutil.copy(REPO / "1_input"      / "polymer_drug_solvate_ion.psf", cdir)

    return gdir


def write_psf(child: Path, combo: Tuple[int, int, int, int]) -> None:
    """
    Rewrite the PSF in *child* folder with chosen bead indices.

    combo = (c_idx, n_idx, qp_idx, qm_idx) with indices 0-5
    Very simple replacement:  C1->C{c}, N1->N{n}, Q1->Q{qp}, Q2->Q{qm}
    Assumes original PSF uses C1/N1 for polymer, Q1/Q2 for +/- charges.
    """
    c_idx, n_idx, qp_idx, qm_idx = combo
    txt = (child / "polymer_drug_solvate_ion.psf").read_text()
    txt = txt.replace(" C1 ", f" {C_LEVELS[c_idx]} ")
    txt = txt.replace(" N1 ", f" {N_LEVELS[n_idx]} ")
    txt = txt.replace(" Q1 ", f" {Q_LEVELS[qp_idx]} ")
    txt = txt.replace(" Q2 ", f" {Q_LEVELS[qm_idx]} ")
    (child / "polymer_drug_solvate_ion.psf").write_text(txt)


def write_slurm(gen: int) -> None:
    script = REPO / "2_scripts" / "submit.sh"
    base   = BASE / "3_simulations" / f"generation_{gen:02d}"
    sims   = " ".join(f"child_{i:02d}" for i in range(1, POP + 1))
    script.write_text(f"""#!/bin/bash
#SBATCH -J ABPO
#SBATCH -A loni_pdrug
#SBATCH -p workq
#SBATCH -N 3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o {base}/out_%j.txt
#SBATCH -e {base}/err_%j.txt
#SBATCH --distribution=block:block

NAMD={NAMD}
BASE={base}
SIMS=({sims})

for s in "${{SIMS[@]}}"; do
  conf="$BASE/$s/config.namd"
  log="$BASE/$s/output.txt"
  srun -N1 --ntasks=1 --cpus-per-task=16 --exclusive "$NAMD" +p16 "$conf" >"$log" 2>&1 &
done
wait
""")
    script.chmod(0o755)


def submit_and_wait() -> None:
    remote = BASE / "2_scripts" / "submit.sh"
    job_id = _ssh_exec(f"sbatch {remote}").split()[-1]
    while _ssh_exec(f"squeue -h -j {job_id}"):
        time.sleep(300)


def avg_rg(child: Path) -> float:
    psf = child / "polymer_drug_solvate_ion.psf"
    dcd = child / "run_pr.dcd"
    u = mda.Universe(psf, dcd)
    sel = u.select_atoms("not (name W HOH Q1A Q2A TQ5)")
    vals = []
    for _ in u.trajectory[-100:]:
        com = sel.center_of_mass()
        diff = sel.positions - com
        sq = np.sum(diff ** 2, axis=1)
        vals.append(np.sqrt(np.sum(sel.masses * sq) / np.sum(sel.masses)))
    return float(np.mean(vals))


# ---------------------------------------------------------------------- #
# ---------------------  ABPO CORE DATA STRUCTURES  -------------------- #
# ---------------------------------------------------------------------- #

ProbDict = Dict[str, np.ndarray]              # key -> prob[6]

def uniform_prob() -> ProbDict:
    return {k: np.ones(6) / 6 for k in ("C", "N", "Qplus", "Qminus")}


def orthogonal_seed() -> List[Tuple[int, int, int, int]]:
    """Return 12 combos where every level appears 2× per factor."""
    base = list(range(6))
    design = [
        (i,
         base[(i + 2) % 6],
         base[(i + 4) % 6],
         base[(i + 1) % 6])
        for i in base
    ]
    random.shuffle(design)
    return design[:POP]                       # 12 combos


def draw_batch(prob: ProbDict,
               best_combo: Tuple[int, int, int, int],
               batch_size: int = POP,
               exploit_k: int = EXPLOIT_K) -> List[Tuple[int, int, int, int]]:
    """Return next batch of indices."""
    out: List[Tuple[int, int, int, int]] = [best_combo]

    # (A) exploitation neighbours
    for _ in range(exploit_k - 1):
        c, n, qp, qm = best_combo
        pos = random.randint(0, 3)
        if pos == 0:
            c = np.random.choice(6, p=prob["C"])
        elif pos == 1:
            n = np.random.choice(6, p=prob["N"])
        elif pos == 2:
            qp = np.random.choice(6, p=prob["Qplus"])
        else:
            qm = np.random.choice(6, p=prob["Qminus"])
        out.append((c, n, qp, qm))

    # (B) exploration draws
    while len(out) < batch_size:
        out.append((
            np.random.choice(6, p=prob["C"]),
            np.random.choice(6, p=prob["N"]),
            np.random.choice(6, p=prob["Qplus"]),
            np.random.choice(6, p=prob["Qminus"]),
        ))
    return out


def update_prob(prob: ProbDict,
                batch_records: List[Tuple[Tuple[int, int, int, int], float]],
                top_fraction: float = TOP_FRAC,
                damp: float = DAMP) -> ProbDict:
    """Rank-based probability update."""
    # sort by error (lower = better)
    batch_records.sort(key=lambda x: x[1])
    cutoff = max(1, int(len(batch_records) * top_fraction))
    top = batch_records[:cutoff]
    rest = batch_records[cutoff:]

    delta = {k: np.zeros(6) for k in prob}
    for (c, n, qp, qm), _err in top:
        delta["C"][c] += 1;      delta["N"][n] += 1
        delta["Qplus"][qp] += 1; delta["Qminus"][qm] += 1
    for (c, n, qp, qm), _err in rest:
        delta["C"][c] -= 1;      delta["N"][n] -= 1
        delta["Qplus"][qp] -= 1; delta["Qminus"][qm] -= 1

    for key in prob:
        new = prob[key] + damp * delta[key]
        new[new < 0] = 0.001
        new /= new.sum()
        # occasional reseed to avoid premature collapse
        if random.random() < RESEED_CHANCE:
            new = (new + 1/6) / (1 + 1)
        prob[key] = new
    return prob


# ---------------------------------------------------------------------- #
# ---------------------------  CSV LOGGING  ---------------------------- #
# ---------------------------------------------------------------------- #

def log_csv(csv_path: Path,
            gen: int,
            child: int,
            combo: Tuple[int, int, int, int],
            rg: float,
            err: float) -> None:
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["gen","child","C","N","Qplus","Qminus","Rg","err"])
        c,n,qp,qm = combo
        w.writerow([gen, child,
                    C_LEVELS[c], N_LEVELS[n], Q_LEVELS[qp], Q_LEVELS[qm],
                    f"{rg:.3f}", f"{err:.3f}"])


# ---------------------------------------------------------------------- #
# -----------------------------  DRIVER  ------------------------------- #
# ---------------------------------------------------------------------- #

def in_target_window(rg: float) -> bool:
    return LOWER_RG <= rg <= UPPER_RG


def main() -> None:
    prob = uniform_prob()
    csv_file = REPO / "optimisation_results.csv"
    total_runs = 0
    best_rec: Dict | None = None          # {"combo": ..., "Rg": ..., "err": ...}

    for gen in itertools.count(0):
        # ---------- choose next batch ----------
        if gen == 0:
            batch = orthogonal_seed()
        else:
            batch = draw_batch(prob, best_rec["combo"], POP, exploit_k=EXPLOIT_K)

        # ---------- prepare directories ----------
        gdir = create_generation(gen)
        for idx, combo in enumerate(batch, 1):
            write_psf(gdir / f"child_{idx:02d}", combo)

        write_slurm(gen)
        submit_and_wait()

        # ---------- evaluate ----------
        batch_records = []                        # [(combo, error)]
        for idx, combo in enumerate(batch, 1):
            child = gdir / f"child_{idx:02d}"
            rg   = avg_rg(child)
            err  = abs(rg - TARGET_RG)
            batch_records.append((combo, err))
            log_csv(csv_file, gen, idx, combo, rg, err)
            print(f"G{gen:02d} C{idx:02d}: {combo}  Rg={rg:5.2f}  |Δ|={err:4.2f}")

        # ---------- track best ----------
        batch_best = min(batch_records, key=lambda x: x[1])
        if (best_rec is None) or (batch_best[1] < best_rec["err"]):
            best_rec = {
                "combo": batch_best[0],
                "Rg":    TARGET_RG - batch_best[1]
                         if batch_best[1] < TARGET_RG else TARGET_RG + batch_best[1],
                "err":   batch_best[1],
            }

        total_runs += POP

        # ---------- stopping conditions ----------
        if in_target_window(best_rec["Rg"]):
            print("✅  Target window hit -> stopping.")
            break
        if total_runs >= TOTAL_CAP or gen >= GEN_MAX:
            print("⛔  Budget cap reached. Stopping.")
            break

        # ---------- update probabilities ----------
        prob = update_prob(prob, batch_records)

    # ---------- final report ----------
    c, n, qp, qm = best_rec["combo"]
    print("\n=== FINAL RESULT ===")
    print(f"Best combo:  C={C_LEVELS[c]}  N={N_LEVELS[n]}  "
          f"Q+={Q_LEVELS[qp]}  Q-={Q_LEVELS[qm]}")
    print(f"Rg = {best_rec['Rg']:.3f} Å   |Δ| = {best_rec['err']:.3f} Å")

    if _SSH_CLIENT is not None:
        _SSH_CLIENT.close()


if __name__ == "__main__":
    main()
