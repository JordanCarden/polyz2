import bisect
import csv
import shutil
import time
from pathlib import Path
from typing import List

import cma
import MDAnalysis as mda
import numpy as np
import paramiko


BASE = Path("/work/jcarde7/polyzwitterion")
POP = 12
TARGET_RG = 36.0
GEN_MAX = 30

REPO = Path(__file__).resolve().parent.parent
NAMD = BASE / "NAMD_2.14_Linux-x86_64-multicore/namd2"

LOWER_RG = 35.0
UPPER_RG = 37.0

_SSH_CLIENT: paramiko.SSHClient | None = None


def _get_ssh() -> paramiko.SSHClient:
    """Return a connected singleton SSH client.

    Returns:
        A connected `paramiko.SSHClient`.
    """
    global _SSH_CLIENT
    if _SSH_CLIENT is None:
        client = paramiko.SSHClient()
        client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect("qbd.loni.org", username="jcarde7")
        _SSH_CLIENT = client
    return _SSH_CLIENT


def _ssh_exec(cmd: str) -> str:
    """Execute *cmd* remotely and return stdout.

    Args:
        cmd: Command to run on the HPC login node.

    Returns:
        Standard output with trailing whitespace stripped.
    """
    _, stdout, _ = _get_ssh().exec_command(cmd)
    return stdout.read().decode().strip()


def in_target_window(rg: float) -> bool:
    """Check whether *rg* falls within the desired window.

    Args:
        rg: Radius of gyration.

    Returns:
        True if 35 Å ≤ rg ≤ 37 Å, else False.
    """
    return LOWER_RG <= rg <= UPPER_RG


def load_pairs(nbfix: Path) -> list[tuple[str, str]]:
    """Return nonidentical atom-type pairs from the NBFIX section.

    Args:
        nbfix: Path to ``nbfix.par``.

    Returns:
        Pairs ``(a, b)`` where ``a != b`` in file order.
    """
    pairs: list[tuple[str, str]] = []
    in_section = False
    for line in nbfix.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if not in_section:
            if parts[0] == "NBFIX":
                in_section = True
            continue
        if len(parts) >= 4:
            a, b = parts[0], parts[1]
            if a != b:
                pairs.append((a, b))
    return pairs


def load_allowed() -> list[float]:
    """Load allowed epsilon values from file.

    Returns:
        Sorted absolute values from the parameter file.
    """
    txt = REPO / "0_parameters" / "valid_parameter_values.txt"
    return sorted(abs(float(x)) for x in txt.read_text().split())


PAIRS = load_pairs(REPO / "0_parameters" / "nbfix.par")
N_PAIRS = len(PAIRS)
ALLOWED = load_allowed()
LOW, HIGH = ALLOWED[0], ALLOWED[-1]
INIT_EPS = [np.median(ALLOWED)] * N_PAIRS


def snap_to_allowed(vec: list[float]) -> list[float]:
    """Snap continuous epsilon vector to nearest allowed magnitudes.

    Args:
        vec: Continuous epsilon candidates.

    Returns:
        Vector aligned to discrete allowed magnitudes.
    """
    out = []
    for v in vec:
        mag = abs(v)
        idx = bisect.bisect_left(ALLOWED, mag)
        if idx == 0:
            out.append(ALLOWED[0])
        elif idx == len(ALLOWED):
            out.append(ALLOWED[-1])
        else:
            left, right = ALLOWED[idx - 1], ALLOWED[idx]
            out.append(right if right - mag < mag - left else left)
    return out


def create_generation(gen: int) -> Path:
    """Create directory tree for generation *gen*.

    Args:
        gen: Generation index (1‑based).

    Returns:
        Path to the generation directory.
    """
    gdir = REPO / "3_simulations" / f"generation_{gen:02d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, POP + 1):
        cdir = gdir / f"child_{i:02d}"
        cdir.mkdir(exist_ok=True)
        for src in ("nbfix.par", "config.namd"):
            shutil.copy(REPO / "0_parameters" / src, cdir)
    return gdir


def edit_nbfix(path: Path, eps: list[float]) -> None:
    """Rewrite *path* in‑place with new epsilon values.

    Args:
        path: `nbfix.par` file to edit.
        eps: Epsilon magnitudes for the pair types; length must equal ``len(PAIRS)``.
    """
    if len(eps) != len(PAIRS):
        raise ValueError("epsilon vector length mismatch")
    repl = {p: f"-{v:.8f}" for p, v in zip(PAIRS, eps)}
    out: List[str] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 4 and (parts[0], parts[1]) in repl:
            parts[2] = repl[(parts[0], parts[1])]
            line = "   ".join(parts)
        out.append(line)
    path.write_text("\n".join(out) + "\n")


def write_slurm(gen: int) -> None:
    """Generate a CPU-only SLURM submission script for generation *gen*."""
    script = REPO / "2_scripts" / "submit.sh"
    base = BASE / "3_simulations" / f"generation_{gen:02d}"
    sims = " ".join(f"child_{i:02d}" for i in range(1, POP + 1))
    script.write_text(
        f"""#!/bin/bash
#SBATCH -J XPLODE_HPC
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
  srun -N1 --ntasks=1 --cpus-per-task=16 \\
       --exclusive "$NAMD" +p16 "$conf" >"$log" 2>&1 &
done
wait
"""
    )
    script.chmod(0o755)


def submit_and_wait() -> None:
    """Submit the SLURM job and block until completion."""
    remote = BASE / "2_scripts" / "submit.sh"
    job_id = _ssh_exec(f"sbatch {remote}").split()[-1]
    while _ssh_exec(f"squeue -h -j {job_id}"):
        time.sleep(300)


def avg_rg(child: Path) -> float:
    """Compute mean Rg over the last 100 frames of trajectory.

    Args:
        child: Directory containing `run_pr.dcd`.

    Returns:
        Average radius of gyration in Å.
    """
    psf = REPO / "1_input" / "polymer_drug_solvate_ion.psf"
    dcd = child / "run_pr.dcd"
    u = mda.Universe(psf, dcd)
    sel = u.select_atoms("not (name W HOH Q1A Q2A TQ5)")
    values = []
    for _ in u.trajectory[-100:]:
        com = sel.center_of_mass()
        diff = sel.positions - com
        sq = np.sum(diff**2, axis=1)
        values.append(np.sqrt(np.sum(sel.masses * sq) / np.sum(sel.masses)))
    return float(np.mean(values))


def log_csv(
    path: Path,
    gen: int,
    child: int,
    eps: list[float],
    rg: float,
    err: float,
) -> None:
    """Append one optimisation record to *path*.

    Args:
        path: CSV file.
        gen: Generation index.
        child: Child index.
        eps: Epsilon vector.
        rg: Radius of gyration.
        err: |rg − TARGET_RG|.
    """
    header = ["gen", "child", "eps", "Rg", "err"]
    write_header = not path.exists()
    eps_vec = "[" + " ".join(f"{e:.6f}" for e in eps) + "]"
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow([gen, child, eps_vec, f"{rg:.6f}", f"{err:.6f}"])


def main() -> None:
    """Run CMA‑ES optimisation with early stop on valid Rg window."""
    es = cma.CMAEvolutionStrategy(
        INIT_EPS,
        0.8,
        {"popsize": POP, "bounds": [[LOW] * N_PAIRS, [HIGH] * N_PAIRS]},
    )
    csv_file = REPO / "optimisation_results.csv"
    gen = 1
    found_vec: list[float] | None = None
    found_rg: float | None = None
    found_err: float | None = None

    while not es.stop() and gen <= GEN_MAX:
        gdir = create_generation(gen)
        raw = es.ask()
        vecs = [snap_to_allowed(v) for v in raw]
        for idx, v in enumerate(vecs, 1):
            edit_nbfix(gdir / f"child_{idx:02d}" / "nbfix.par", v)

        write_slurm(gen)
        submit_and_wait()

        fitness: list[float] = []
        hit = False
        for idx in range(1, POP + 1):
            child = gdir / f"child_{idx:02d}"
            rg = avg_rg(child)
            err = abs(rg - TARGET_RG)
            fitness.append(err)
            log_csv(csv_file, gen, idx, vecs[idx - 1], rg, err)
            print(
                f"gen {gen:02d} child {idx:02d}  "
                f"Rg={rg:6.2f}  |Δ|={err:5.2f}  "
                f"eps=[{' '.join(f'{e:.2f}' for e in vecs[idx - 1])}]"
            )
            if in_target_window(rg):
                hit = True
                found_vec = vecs[idx - 1]
                found_rg = rg
                found_err = err

        es.tell(vecs, fitness)
        if hit:
            break
        gen += 1

    if found_vec is not None:
        print(
            "early-stop hit:",
            f"Rg={found_rg:.3f}",
            f"|Δ|={found_err:.3f}",
            f"eps={found_vec}",
        )
    else:
        print("best ε:", es.result.xbest, "fitness:", es.result.fbest)
    if _SSH_CLIENT is not None:
        _SSH_CLIENT.close()


if __name__ == "__main__":
    main()