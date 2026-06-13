"""Reconstruct an N-replica failure storm as an animated GIF, entirely from the run's
JSONL telemetry (chaos.jsonl faults + replica*.jsonl commits/lifecycle). No live
recording needed — the timestamps are the ground truth — so the time axis is ours to
compress (30 min -> ~25 s) and the encoding ours to choose. A grid of replica cells
(one per group) shows each replica's state; a top strip tracks quorum size, cumulative
commits, and the active fault. This is the scale analogue of the 2-worker demo.gif:
the close-up shows the recovery mechanism, this wide shot shows a swarm absorbing chaos.

  python analysis/storm_gif.py --run experiments/storm-n32p --n 32 --out plots/m5_storm_n32.gif \
      --seconds 26 --fps 18
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from storm_events import build, load_jsonl, state

BG = "#0d1117"
FG = "#c9d1d9"
COL = {
    "training": "#3fb950",     # alive, between syncs
    "commit": "#f0f6fc",       # just committed a sync (brief flash)
    "down": "#5a1e1e",         # killed, awaiting supervisor relaunch
    "stopped": "#d29922",      # SIGSTOP straggler
    "partition": "#388bfd",    # link down
    "recover": "#a371f7",      # relaunched, healing (no commit yet)
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--seconds", type=float, default=26.0)
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--cols", type=int, default=8)
    p.add_argument("--pulse", type=float, default=4.0, help="wall-seconds a commit flash lasts")
    args = p.parse_args()
    run = Path(args.run)
    n = args.n
    starts, commits, kills, stops, parts, quorum, faults_tl = build(run, n)

    # time span: chaos window padded, fall back to commit span
    chaos = load_jsonl(run / "chaos.jsonl")
    cs = [e["ts"] for e in chaos if e.get("event") == "chaos_start"]
    ce = [e["ts"] for e in chaos if e.get("event") == "chaos_end"]
    all_c = [c for r in range(n) for c in commits[r]]
    t0 = (cs[0] - 15) if cs else (min(all_c) if all_c else 0)
    t1 = (ce[0] + 8) if ce else (max(all_c) if all_c else 1)
    nframes = int(args.seconds * args.fps)
    qts = [q[0] for q in quorum]
    qsz = [q[1] for q in quorum]

    cols = args.cols
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(9.0, 6.4), facecolor=BG)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.6], hspace=0.28,
                          left=0.07, right=0.97, top=0.93, bottom=0.06)
    axq = fig.add_subplot(gs[0]); axg = fig.add_subplot(gs[1])
    for ax in (axq, axg):
        ax.set_facecolor(BG)
        for s in ax.spines.values():
            s.set_visible(False)

    # quorum timeline (static faint full line + animated fill/marker)
    axq.plot(qts, qsz, color="#30363d", lw=1.0)
    axq.set_xlim(t0, t1); axq.set_ylim(0, n + 1)
    axq.set_ylabel("quorum", color=FG, fontsize=9)
    axq.tick_params(colors="#6e7681", labelsize=7)
    axq.set_xticks([])
    axq.axhline(n, color="#21262d", lw=0.8, ls=":")
    fill = axq.fill_between([t0, t0], [0, 0], color="#1f6feb", alpha=0.25)
    qmark, = axq.plot([], [], "o", color="#58a6ff", ms=5)
    title = axq.set_title("", color=FG, fontsize=12, loc="left", pad=8)

    # grid of replica cells
    axg.set_xlim(0, cols); axg.set_ylim(0, rows); axg.invert_yaxis()
    axg.set_xticks([]); axg.set_yticks([]); axg.set_aspect("equal")
    rects, labels = [], []
    for r in range(n):
        cx, cy = r % cols, r // cols
        rect = Rectangle((cx + 0.06, cy + 0.06), 0.88, 0.88, facecolor=COL["down"],
                         edgecolor="#0d1117", lw=1.5)
        axg.add_patch(rect); rects.append(rect)
        labels.append(axg.text(cx + 0.5, cy + 0.5, str(r), ha="center", va="center",
                               color="#0d1117", fontsize=8, fontweight="bold"))
    ticker = axg.text(0.0, -0.28, "", color="#f85149", fontsize=11,
                      fontweight="bold", va="bottom")
    legend = axg.text(cols, -0.28, "", color="#8b949e", fontsize=7.5, va="bottom", ha="right")
    legend.set_text("training  commit  straggler  partition  healing  down")

    def update(i):
        t = t0 + (t1 - t0) * i / max(1, nframes - 1)
        ncommit = sum(1 for r in range(n) for c in commits[r] if c <= t)
        live_q = [s for s, _ in quorum if s <= t]
        cur_q = qsz[len(live_q) - 1] if live_q else 0
        for r in range(n):
            st = state(r, t, starts, commits, kills, stops, parts)
            if st == "training" and any(t - args.pulse <= c <= t for c in commits[r]):
                st = "commit"
            rects[r].set_facecolor(COL[st])
            labels[r].set_color("#0d1117" if st in ("commit", "training") else "#f0f6fc")
        nonlocal fill
        fill.remove()
        seg_t = [s for s in qts if s <= t] or [t0]
        seg_q = qsz[:len(seg_t)] or [0]
        fill = axq.fill_between(seg_t, seg_q, color="#1f6feb", alpha=0.22)
        qmark.set_data([t], [cur_q])
        mins = (t - t0) / 60.0
        speed = (t1 - t0) / args.seconds
        title.set_text(f"32-replica DiLoCo failure storm   t+{mins:4.1f} min   "
                       f"quorum {cur_q}/{n}   commits {ncommit}   ({speed:.0f}x)")
        recent = [lab for (ts, lab) in faults_tl if t - args.pulse * 1.5 <= ts <= t]
        ticker.set_text("⚡ " + "   ".join(recent[-3:]) if recent else "")
        return rects + labels + [qmark, title, ticker]

    anim = FuncAnimation(fig, update, frames=nframes, blit=False)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(args.out, writer=PillowWriter(fps=args.fps))
    print(f"wrote {args.out}  ({nframes} frames, {args.seconds}s @ {args.fps}fps)")


if __name__ == "__main__":
    main()
