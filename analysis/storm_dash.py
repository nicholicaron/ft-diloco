"""Live, scalable storm dashboard. Unlike the old per-worker-panel dashboard.py (which
stacks one panel per replica and falls apart past a handful), this is AGGREGATE and
fixed-size at any N: a colored replica-state grid (auto-laid-out), a quorum-size
sparkline, cluster metrics, and a fault feed — refreshed ~1/s from the same JSONL the
storm writes. Run it on the storm host while a storm is in flight, or with --once for a
single frame. Screen-record it if you want a live-capture GIF; otherwise storm_gif.py
reconstructs the same picture post-hoc.

  python analysis/storm_dash.py --run experiments/storm-n32p --n 32        # live
  python analysis/storm_dash.py --run experiments/storm-n32p --n 32 --once # one frame
"""

import argparse
import time
from pathlib import Path

from rich.align import Align
from rich.console import Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from storm_events import build, load_jsonl, state

HEX = {
    "training": "#1f6f33", "commit": "#f0f6fc", "down": "#7a1f1f",
    "stopped": "#9e7016", "partition": "#1f6feb", "recover": "#6e40c9",
}
FG = {"commit": "#0d1117", "training": "#d6f5d6"}
SPARK = " ▁▂▃▄▅▆▇█"


def spark(vals, n):
    if not vals:
        return ""
    return "".join(SPARK[min(len(SPARK) - 1, round(v / max(1, n) * (len(SPARK) - 1)))] for v in vals)


def frame(run: Path, n: int, cols: int, t0_ref):
    starts, commits, kills, stops, parts, quorum, faults_tl = build(run, n)
    now = time.time()
    counts = dict.fromkeys(HEX, 0)
    grid = Table.grid(padding=(0, 1))
    for _ in range(cols):
        grid.add_column(justify="center")
    cells = []
    for r in range(n):
        st = state(r, now, starts, commits, kills, stops, parts)
        if st == "training" and any(now - 1.5 <= c <= now for c in commits[r]):
            st = "commit"
        counts[st] += 1
        fg = FG.get(st, "#e6edf3")
        cells.append(Text(f" {r:>2} ", style=f"bold {fg} on {HEX[st]}"))
    for i in range(0, n, cols):
        grid.add_row(*cells[i:i + cols])

    live = counts["training"] + counts["commit"] + counts["recover"]
    last_q = quorum[-1][1] if quorum else 0
    ncommit = sum(len(commits[r]) for r in range(n))
    recent = [q for ts, q in quorum if now - 90 <= ts]
    rate = sum(1 for r in range(n) for c in commits[r] if now - 60 <= c <= now) / 60.0
    elapsed = (now - t0_ref) / 60.0 if t0_ref else 0.0

    meta = Table.grid(padding=(0, 2))
    meta.add_column(justify="right", style="bold #58a6ff")
    meta.add_column()
    meta.add_row("replicas", f"{n}   live(not faulted) [bold]{live}[/]")
    meta.add_row("last quorum", f"[bold]{last_q}[/]/{n}    sparkline {spark(recent[-40:], n)}")
    meta.add_row("commits", f"{ncommit}   (~{rate:.2f}/s, last 60s)")
    meta.add_row("state", " ".join(f"[on {HEX[k]}] {counts[k]} [/]{k}" for k in
                                    ("training", "down", "stopped", "partition", "recover")))
    meta.add_row("elapsed", f"{elapsed:.1f} min")

    feed_lines = []
    for ts, lab in faults_tl[-7:]:
        ago = now - ts
        glyph = "kill" if lab.startswith("kill") else lab.split()[0]
        col = {"kill": "#f85149", "straggler": "#d29922", "partition": "#1f6feb"}.get(glyph, "#8b949e")
        feed_lines.append(Text(f"  t-{ago:4.0f}s  ", style="#6e7681") + Text(lab, style=f"bold {col}"))
    feed = Group(*feed_lines) if feed_lines else Text("  (no faults yet)", style="#6e7681")

    body = Group(
        Align.center(grid),
        Text(""),
        meta,
        Text("recent faults", style="bold #6e7681"),
        feed,
    )
    return Panel(body, title=f"[bold]ft-diloco failure storm — {run.name}[/]",
                 border_style="#1f6feb", padding=(1, 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--once", action="store_true")
    p.add_argument("--interval", type=float, default=1.0)
    args = p.parse_args()
    run = Path(args.run)
    # reference t0 = earliest start event, for elapsed display
    t0s = [e["ts"] for f in run.glob("replica*.jsonl") for e in load_jsonl(f) if e.get("phase") == "start"]
    t0 = min(t0s) if t0s else None

    if args.once:
        from rich.console import Console
        Console().print(frame(run, args.n, args.cols, t0))
        return
    with Live(frame(run, args.n, args.cols, t0), refresh_per_second=4, screen=True) as live:
        while True:
            time.sleep(args.interval)
            live.update(frame(run, args.n, args.cols, t0))


if __name__ == "__main__":
    main()
