"""Shared storm event-derivation: turn a run's JSONL (chaos.jsonl faults +
replica*.jsonl commits/lifecycle) into per-replica state over time. Used by both the
post-hoc GIF (storm_gif.py) and the live dashboard (storm_dash.py) so they agree on
what "down / straggler / partition / healing / training / commit" means. Pure stdlib —
no matplotlib — so the live dashboard stays light.
"""

import json
from pathlib import Path

STATES = ("training", "commit", "down", "stopped", "partition", "recover")


def load_jsonl(path: Path):
    out = []
    if path.exists():
        for line in path.open():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def build(run: Path, n: int):
    """Return per-replica event lists + the quorum-size time series + a fault ticker."""
    starts = {r: [] for r in range(n)}
    commits = {r: [] for r in range(n)}
    quorum = []  # (ts, num_participants)
    for f in run.glob("replica*.jsonl"):
        try:
            r = int(f.stem.replace("replica", ""))
        except ValueError:
            continue
        if r >= n:
            continue
        for e in load_jsonl(f):
            if e.get("phase") == "start":
                starts[r].append(e["ts"])
            elif e.get("event") == "outer_sync" and e.get("committed"):
                commits[r].append(e["ts"])
                quorum.append((e["ts"], e["num_participants"]))
    for r in range(n):
        starts[r].sort()
        commits[r].sort()
    quorum.sort()

    kills = {r: [] for r in range(n)}
    stops = {r: [] for r in range(n)}      # (stop_ts, cont_ts)
    parts = {r: [] for r in range(n)}      # (partition_ts, heal_ts)
    faults_tl = []                         # (ts, label)
    open_stop, open_part = {}, {}
    for e in load_jsonl(run / "chaos.jsonl"):
        if e.get("event") != "fault" or not e.get("ok"):
            continue
        r, fault, ts = e.get("target"), e["fault"], e["ts"]
        skipped = isinstance(e.get("result"), dict) and "skipped" in e["result"]
        if r is None or r >= n:
            continue
        if fault in ("kill", "kill_safe") and not skipped:
            kills[r].append(ts)
            faults_tl.append((ts, f"kill r{r}"))
        elif fault == "stop":
            open_stop[r] = ts
            faults_tl.append((ts, f"straggler r{r}"))
        elif fault == "cont" and r in open_stop:
            stops[r].append((open_stop.pop(r), ts))
        elif fault == "partition":
            open_part[r] = ts
            faults_tl.append((ts, f"partition r{r}"))
        elif fault == "heal" and r in open_part:
            parts[r].append((open_part.pop(r), ts))
    big = (quorum[-1][0] if quorum else 0) + 1e9
    for r, ts in open_stop.items():
        stops[r].append((ts, big))
    for r, ts in open_part.items():
        parts[r].append((ts, big))
    faults_tl.sort()
    return starts, commits, kills, stops, parts, quorum, faults_tl


def state(r, t, starts, commits, kills, stops, parts):
    """State of replica r at wall-time t (excludes the brief commit flash — callers
    overlay that from `commits` with their own pulse window)."""
    for a, b in parts[r]:
        if a <= t <= b:
            return "partition"
    for a, b in stops[r]:
        if a <= t <= b:
            return "stopped"
    kb = [k for k in kills[r] if k <= t]
    if kb and not any(max(kb) < s <= t for s in starts[r]):
        return "down"
    sb = [s for s in starts[r] if s <= t]
    if not sb:
        return "down"
    last_start = max(sb)
    if not any(c > last_start and c <= t for c in commits[r]):
        return "recover"
    return "training"
