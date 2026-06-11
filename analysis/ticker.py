"""Live terminal ticker for the demo recording. Tails a replica's JSONL and renders
status + loss sparkline; or renders a cluster summary across replicas.

  python analysis/ticker.py --run experiments/m2-gif --replica 0 --name "worker 0"
  python analysis/ticker.py --run experiments/m2-gif --cluster
"""

import argparse
import json
import time
from pathlib import Path

RESET, BOLD, DIM = "\033[0m", "\033[1m", "\033[2m"
GREEN, RED, YELLOW, CYAN = "\033[32m", "\033[31m", "\033[33m", "\033[36m"
BLOCKS = "▁▂▃▄▅▆▇█"


def load(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
    for line in path.open():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def sparkline(vals: list[float], width: int = 48) -> str:
    if len(vals) < 2:
        return ""
    vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    return "".join(BLOCKS[int((v - lo) / rng * (len(BLOCKS) - 1))] for v in vals)


def replica_frame(run: Path, rid: int, name: str) -> str:
    evs = load(run / f"replica{rid}.jsonl")
    steps = [e for e in evs if e["event"] == "step"]
    syncs = [e for e in evs if e["event"] == "outer_sync"]
    starts = [e for e in evs if e.get("phase") == "start"]
    now = time.time()

    if not steps:
        status, color = "STARTING", YELLOW
    else:
        age = now - steps[-1]["ts"]
        recent_start = starts and (now - starts[-1]["ts"]) < 150
        if age < 15:
            status, color = "TRAINING", GREEN
        elif recent_start:
            # post-relaunch stall = quorum join + P2P state transfer in progress
            status, color = "RECOVERING", YELLOW
        else:
            status, color = "DEAD", RED

    lines = [f"{BOLD}{name}{RESET}   {color}{BOLD}● {status}{RESET}"]
    if steps:
        s = steps[-1]
        lines.append(
            f"step {BOLD}{s['step']:>5}{RESET}   loss {BOLD}{s['loss']:.3f}{RESET}"
            f"   {s['tokens'] / 1e6:.1f}M tokens   pid {starts[-1]['pid'] if starts else '?'}"
        )
        lines.append(CYAN + sparkline([e["loss"] for e in steps]) + RESET)
    if syncs:
        y = syncs[-1]
        lines.append(
            f"{DIM}outer step {y['outer_step']}  participants {y['num_participants']}"
            f"  committed {y['committed']}{RESET}"
        )
    return "\n".join(lines)


def cluster_frame(run: Path) -> str:
    lines = [f"{BOLD}cluster — {run.name}{RESET}"]
    total_committed = 0
    parts = "?"
    for f in sorted(run.glob("replica*.jsonl")):
        rid = int(f.stem.replace("replica", ""))
        syncs = [e for e in load(f) if e["event"] == "outer_sync"]
        committed = sum(1 for e in syncs if e["committed"])
        total_committed += committed
        if syncs:
            parts = syncs[-1]["num_participants"]
        lines.append(f"  worker {rid}: {committed} committed syncs")
    lines.append(f"{BOLD}participants: {GREEN}{parts}{RESET}   total commits: {total_committed}")
    chaos = [e for e in load(run / "chaos.jsonl") if e.get("event") == "fault"]
    for c in chaos[-4:]:
        lines.append(f"{RED}⚡ {c['fault']} → worker {c['target']}{RESET}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--replica", type=int)
    p.add_argument("--name", default=None)
    p.add_argument("--cluster", action="store_true")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    run = Path(args.run)
    name = args.name or (f"worker {args.replica}" if args.replica is not None else "")

    while True:
        frame = cluster_frame(run) if args.cluster else replica_frame(run, args.replica, name)
        print("\033[2J\033[H" + frame, flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
