#!/usr/bin/env python3
from __future__ import annotations
import csv, itertools, random, shutil, time, re, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import MDAnalysis as mda
import numpy as np
import paramiko

# -------- constants --------
BASE        = Path("/work/jcarde7/polyz2")
POP         = 12
TARGET_RG   = 36.0
LOWER_RG, UPPER_RG = 35.0, 37.0
GEN_MAX, TOTAL_CAP = 10, 120

REPO  = Path(__file__).resolve().parent.parent
NAMD  = BASE / "NAMD_2.14_Linux-x86_64-multicore/namd2"

C_LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6"]
N_LEVELS = ["N1", "N2", "N3", "N4", "N5", "N6"]
Q_LEVELS = ["Q1", "Q2", "Q3", "Q4", "Q5"]

C_SIZE, N_SIZE, Q_SIZE = len(C_LEVELS), len(N_LEVELS), len(Q_LEVELS)

DAMP, TOP_FRAC, EXPLOIT_K, RESEED_CHANCE = 0.2, 0.25, 8, 0.05

# segmented-run knobs
FRAMES_PER_SEGMENT = 100     # you asked to run 100 frames per segment
SEGMENTS            = 4       # 4 segments => 400 frames total
FINAL_OUTPUT_NAME   = "run_pr"
SEG_OUTPUT_FMT      = "run_pr_s{seg}"   # segment 1..SEGMENTS-1; last writes FINAL_OUTPUT_NAME

# -------- ssh helpers --------
_SSH_CLIENT: paramiko.SSHClient | None = None
def _get_ssh() -> paramiko.SSHClient:
    global _SSH_CLIENT
    if _SSH_CLIENT is None:
        c = paramiko.SSHClient(); c.load_system_host_keys()
        c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        c.connect("qbd.loni.org", username="jcarde7")
        _SSH_CLIENT = c
    return _SSH_CLIENT

def _ssh_exec(cmd: str) -> str:
    _, s, _ = _get_ssh().exec_command(cmd)
    return s.read().decode().strip()

# -------- tiny parsers for template knobs --------
_INT_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s+(\S+)", re.IGNORECASE)

def _parse_int_param(txt: str, key: str) -> Optional[int]:
    # finds lines like "dcdfreq 5000" (ignores comments)
    out = None
    for line in txt.splitlines():
        line = line.split("#", 1)[0]
        m = _INT_RE.match(line)
        if not m: 
            continue
        k, v = m.group(1).lower(), m.group(2)
        if k == key.lower():
            try:
                out = int(v)
            except ValueError:
                pass
    return out

# -------- config editing helpers --------
def _sub_one(regex: str, repl: str, txt: str) -> str:
    return re.sub(regex, repl, txt, count=1, flags=re.IGNORECASE|re.MULTILINE)

def _sub_all(regex: str, repl: str, txt: str) -> str:
    return re.sub(regex, repl, txt, flags=re.IGNORECASE|re.MULTILINE)

def _ensure_line(prefix: str, value: str, txt: str) -> str:
    # ensures a single active line "prefix value" exists (adds if missing; de-duplicates)
    pat = rf"^\s*#?\s*{re.escape(prefix)}\s+.*$"
    if re.search(pat, txt, flags=re.IGNORECASE|re.MULTILINE):
        txt = _sub_all(pat, f"{prefix} {value}", txt)
    else:
        txt = txt.rstrip() + f"\n{prefix} {value}\n"
    return txt

def make_segment_config(base_text: str,
                        seg_idx: int,
                        seg_steps: int,
                        prev_prefix: Optional[str],
                        out_prefix: str,
                        first_step: int,
                        is_first: bool,
                        is_last: bool) -> str:
    txt = base_text

    # output name
    txt = _ensure_line("outputName", out_prefix, txt)

    # run steps
    txt = _ensure_line("run", str(seg_steps), txt)

    # firsttimestep (bookkeeping continuity)
    txt = _ensure_line("firsttimestep", str(first_step), txt)

    if is_first:
        # first segment keeps temperature/minimize settings as in template
        # but ensure any restart inputs are commented out
        txt = _sub_all(r"^\s*bincoordinates\s+.*$", r"# \g<0>", txt)
        txt = _sub_all(r"^\s*binvelocities\s+.*$",  r"# \g<0>", txt)
        txt = _sub_all(r"^\s*extendedSystem\s+.*$", r"# \g<0>", txt)
    else:
        # continuation: use restart files from previous segment
        assert prev_prefix is not None
        txt = _ensure_line("bincoordinates", f"{prev_prefix}.restart.coor", txt)
        txt = _ensure_line("binvelocities",  f"{prev_prefix}.restart.vel",  txt)
        txt = _ensure_line("extendedSystem", f"{prev_prefix}.restart.xsc",  txt)
        # kill temperature initialization and any explicit minimization
        txt = _sub_all(r"^\s*temperature\s+.*$", r"# \g<0>", txt)
        txt = _sub_all(r"^\s*minimi[sz]e\s+.*$", r"# \g<0>", txt)

    return txt

def build_segments_for_child(child_dir: Path) -> None:
    template = child_dir / "config.namd"
    if not template.exists():
        raise FileNotFoundError(f"missing config.namd in {child_dir}")

    base_txt = template.read_text()

    # derive dcdfreq & restartfreq from template (fallbacks if absent)
    dcd_freq = _parse_int_param(base_txt, "dcdfreq") or 5000
    restart_freq = _parse_int_param(base_txt, "restartfreq") or dcd_freq

    steps_per_seg = FRAMES_PER_SEGMENT * dcd_freq
    # nudge to a multiple of restartfreq so segment ends with a checkpoint
    if steps_per_seg % restart_freq != 0:
        steps_per_seg += (restart_freq - (steps_per_seg % restart_freq))

    # write seg configs
    for i in range(1, SEGMENTS + 1):
        is_first = (i == 1)
        is_last  = (i == SEGMENTS)
        out_prefix = FINAL_OUTPUT_NAME if is_last else SEG_OUTPUT_FMT.format(seg=i)
        prev_prefix = None if is_first else (FINAL_OUTPUT_NAME if is_last else SEG_OUTPUT_FMT.format(seg=i-1))
        first_step = (i - 1) * steps_per_seg
        seg_txt = make_segment_config(base_txt, i, steps_per_seg, prev_prefix, out_prefix, first_step, is_first, is_last)
        (child_dir / f"config_seg{i}.namd").write_text(seg_txt)

    # per-child runner (sequential segments on same allocation)
    runner = child_dir / "run_all_segs.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")\"",
        f"NAMD_BIN=\"{NAMD}\"",
        "CORES=${NAMD_CORES:-16}",
    ]
    for i in range(1, SEGMENTS + 1):
        lines.append(f"echo \"[{child_dir.name}] segment {i}/{SEGMENTS} start\"")
        lines.append(f"\"$NAMD_BIN\" +p${{CORES}} config_seg{i}.namd > segment{i}.out 2>&1")
        lines.append(f"echo \"[{child_dir.name}] segment {i}/{SEGMENTS} done\"")
    runner.write_text("\n".join(lines) + "\n")
    os.chmod(runner, (runner.stat().st_mode | 0o111))

# -------- file helpers --------
def create_generation(gen: int) -> Path:
    gdir = REPO / "3_simulations" / f"generation_{gen:02d}"
    gdir.mkdir(parents=True, exist_ok=True)
    for i in range(1, POP + 1):
        cdir = gdir / f"child_{i:02d}"
        cdir.mkdir(exist_ok=True)
        shutil.copy(REPO / "0_parameters" / "config.namd", cdir)
        shutil.copy(REPO / "1_input" / "polymer_drug_solvate_ion.psf", cdir)
        # build per-segment configs + runner for this child
        build_segments_for_child(cdir)
    return gdir

def write_psf(child: Path, combo: Tuple[int, int, int, int]) -> None:
    c, n, qp, qm = combo
    txt = (child / "polymer_drug_solvate_ion.psf").read_text()
    txt = (txt.replace(" C1 ", f" {C_LEVELS[c]} ")
               .replace(" N1 ", f" {N_LEVELS[n]} ")
               .replace(" Q1 ", f" {Q_LEVELS[qp]} ")
               .replace(" Q2 ", f" {Q_LEVELS[qm]} "))
    (child / "polymer_drug_solvate_ion.psf").write_text(txt)

def write_slurm(gen: int) -> None:
    script = REPO / "2_scripts" / "submit.sh"
    base   = BASE / "3_simulations" / f"generation_{gen:02d}"
    sims   = " ".join(f"child_{i:02d}" for i in range(1, POP + 1))
    script.write_text(f"""#!/bin/bash
#SBATCH -J ABPO
#SBATCH -A loni_pdrug -p workq -N 3
#SBATCH --ntasks-per-node=4 --cpus-per-task=16
#SBATCH -t 72:00:00
#SBATCH -o {base}/out_%j.txt -e {base}/err_%j.txt
NAMD={NAMD}; BASE={base}; SIMS=({sims})
for s in "${{SIMS[@]}}"; do
  srun -N1 --ntasks=1 --cpus-per-task=16 --exclusive \\
       bash "$BASE/$s/run_all_segs.sh" >"$BASE/$s/output.txt" 2>&1 &
done
wait
""")
    script.chmod(0o755)

def submit_and_wait() -> None:
    jid = _ssh_exec(f"sbatch {BASE / '2_scripts' / 'submit.sh'}").split()[-1]
    while _ssh_exec(f"squeue -h -j {jid}"):
        time.sleep(300)

def avg_rg(child: Path) -> float:
    u = mda.Universe(child / "polymer_drug_solvate_ion.psf",
                     child / f"{FINAL_OUTPUT_NAME}.dcd")
    sel = u.select_atoms("not (name W Q1A Q2A)")
    vals = []
    for _ in u.trajectory[-50:]:
        com = sel.center_of_mass()
        vals.append(
            np.sqrt(((sel.positions - com)**2).sum(1).dot(sel.masses) /
                    sel.masses.sum())
        )
    return float(np.mean(vals))

# -------- probability utils --------
ProbDict = Dict[str, np.ndarray]

def uniform_prob() -> ProbDict:
    return {
        "C":      np.ones(C_SIZE) / C_SIZE,
        "N":      np.ones(N_SIZE) / N_SIZE,
        "Qplus":  np.ones(Q_SIZE) / Q_SIZE,
        "Qminus": np.ones(Q_SIZE) / Q_SIZE,
    }

def orthogonal_seed() -> List[Tuple[int, int, int, int]]:
    all_combos = list(itertools.product(range(C_SIZE),
                                        range(N_SIZE),
                                        range(Q_SIZE),
                                        range(Q_SIZE)))
    random.shuffle(all_combos)
    return all_combos[:POP]

def draw_batch(prob: ProbDict,
               best: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    out = [best]
    for _ in range(EXPLOIT_K - 1):
        c, n, qp, qm = best
        pos = random.randint(0, 3)
        if pos == 0:
            c = np.random.choice(C_SIZE, p=prob["C"])
        elif pos == 1:
            n = np.random.choice(N_SIZE, p=prob["N"])
        elif pos == 2:
            qp = np.random.choice(Q_SIZE, p=prob["Qplus"])
        else:
            qm = np.random.choice(Q_SIZE, p=prob["Qminus"])
        out.append((c, n, qp, qm))

    while len(out) < POP:
        out.append((
            np.random.choice(C_SIZE, p=prob["C"]),
            np.random.choice(N_SIZE, p=prob["N"]),
            np.random.choice(Q_SIZE, p=prob["Qplus"]),
            np.random.choice(Q_SIZE, p=prob["Qminus"]),
        ))
    return out

def update_prob(prob: ProbDict,
                recs: List[Tuple[Tuple[int, int, int, int], float]]) -> ProbDict:
    recs.sort(key=lambda x: x[1])
    cut = max(1, int(len(recs) * TOP_FRAC))

    delta = {k: np.zeros_like(prob[k]) for k in prob}

    for (c, n, qp, qm), _ in recs[:cut]:
        delta["C"][c]      += 1
        delta["N"][n]      += 1
        delta["Qplus"][qp] += 1
        delta["Qminus"][qm] += 1

    for (c, n, qp, qm), _ in recs[cut:]:
        delta["C"][c]      -= 1
        delta["N"][n]      -= 1
        delta["Qplus"][qp] -= 1
        delta["Qminus"][qm] -= 1

    for k in prob:
        p = prob[k] + DAMP * delta[k]
        p[p < 0] = 0.001
        prob[k] = p / p.sum()
        if random.random() < RESEED_CHANCE:
            prob[k] = (prob[k] + 1 / len(prob[k])) / (1 + 1)

    return prob

# -------- logging --------
def log_csv(path: Path, gen: int, child: int, combo, rg, err) -> None:
    hdr = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.writer(f)
        if hdr:
            w.writerow(["gen", "child", "C", "N", "Qplus", "Qminus", "Rg", "err"])
        c, n, qp, qm = combo
        w.writerow([
            gen, child,
            C_LEVELS[c], N_LEVELS[n], Q_LEVELS[qp], Q_LEVELS[qm],
            f"{rg:.3f}", f"{err:.3f}"
        ])

# -------- main loop --------
def in_window(r: float) -> bool:
    return LOWER_RG <= r <= UPPER_RG

def main() -> None:
    prob       = uniform_prob()
    csv_file   = REPO / "optimisation_results.csv"
    total_runs = 0
    best       = None

    for gen in itertools.count(0):
        batch = orthogonal_seed() if gen == 0 else draw_batch(prob, best["combo"])
        gdir  = create_generation(gen)

        for i, combo in enumerate(batch, 1):
            write_psf(gdir / f"child_{i:02d}", combo)

        write_slurm(gen)
        submit_and_wait()

        recs = []
        for i, combo in enumerate(batch, 1):
            rg  = avg_rg(gdir / f"child_{i:02d}")
            err = abs(rg - TARGET_RG)
            recs.append((combo, err))
            log_csv(csv_file, gen, i, combo, rg, err)
            print(f"G{gen:02d} C{i:02d}: {combo}  Rg={rg:5.2f}  |Δ|={err:4.2f}")

        combo_best, err_best = min(recs, key=lambda x: x[1])
        if (best is None) or (err_best < best["err"]):
            # reconstruct best Rg from target ± error for the in_window check
            rg_best = TARGET_RG - err_best if (recs[recs.index((combo_best, err_best))][1] < 0) else TARGET_RG + err_best
            best = {"combo": combo_best, "Rg": rg_best, "err": err_best}

        total_runs += POP

        if in_window(best["Rg"]):
            break
        if total_runs >= TOTAL_CAP or gen >= GEN_MAX:
            break

        prob = update_prob(prob, recs)

    c, n, qp, qm = best["combo"]
    print(f"\nBest: C={C_LEVELS[c]} N={N_LEVELS[n]} "
          f"Q+={Q_LEVELS[qp]} Q-={Q_LEVELS[qm]}  "
          f"Rg={best['Rg']:.3f} Å |Δ|={best['err']:.3f} Å")

    if _SSH_CLIENT:
        _SSH_CLIENT.close()

if __name__ == "__main__":
    main()
