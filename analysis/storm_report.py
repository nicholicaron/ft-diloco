"""Headline report for an N-way storm: step efficiency vs a fault-free reference,
quorum-size distribution, fault tally, and recovery latencies. One number-dense block,
reusable across N. (A file, not an inline -c, to dodge the nested-quoting footgun.)

  python analysis/storm_report.py --run experiments/storm-n32 --reference experiments/storm-n32-ref --h 20
"""

import argparse
import statistics as st
from collections import Counter
from pathlib import Path

from parse_logs import fuse, load_jsonl
from plot_storm import chaos_window, committed_rate


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--h", type=int, default=20)
    args = p.parse_args()
    run, ref = Path(args.run), Path(args.reference)

    win = chaos_window(run)
    wall = win[1] - win[0]
    srate, sn = committed_rate(run, args.h, win)
    rrate, _ = committed_rate(ref, args.h, chaos_window(ref))

    print(f"=== {run.name} : storm vs {ref.name} ===")
    print(f"chaos window         {wall:.0f}s")
    print(f"fault-free ref rate  {rrate:.1f} committed steps/s")
    print(f"storm rate           {srate:.1f} committed steps/s ({sn} committed syncs)")
    print(f"STEP EFFICIENCY      {srate / rrate:.1%}")

    # quorum size over committed syncs in window
    parts = []
    for f in run.glob("replica*.jsonl"):
        for e in load_jsonl(f):
            if (e.get("event") == "outer_sync" and e.get("committed")
                    and win[0] <= e["ts"] <= win[1]):
                parts.append(e["num_participants"])
    if parts:
        print(f"quorum size          min={min(parts)} median={int(st.median(parts))} "
              f"mean={st.mean(parts):.1f} max={max(parts)}  (over {len(parts)} commits)")

    # faults
    raw = [e for e in load_jsonl(run / "chaos.jsonl") if e.get("event") == "fault"]
    ex = [e for e in raw if e.get("ok")
          and not (isinstance(e.get("result"), dict) and "skipped" in e["result"])]
    sk = [e for e in raw if isinstance(e.get("result"), dict) and "skipped" in e["result"]]
    print(f"faults executed      {len(ex)} {dict(Counter(e['fault'] for e in ex))} "
          f"| skipped(no-donor)={len(sk)} | rate={len(ex)/(wall/3600):.0f}/hr")

    # recovery latencies + commit rate
    d = fuse(run)
    ks = [f for f in d["faults"] if f["fault"] == "kill_safe" and "t_back_s" in f]
    tb = sorted(f["t_back_s"] for f in ks)
    tr = sorted(f["t_resume_s"] for f in d["faults"] if "t_resume_s" in f)
    if tb:
        print(f"kills full-recovered {len(ks)}  t_back median={st.median(tb):.1f}s "
              f"p90={tb[min(int(.9 * len(tb)), len(tb) - 1)]:.1f}s max={tb[-1]:.1f}s")
    if tr:
        print(f"survivor t_resume    median={st.median(tr):.1f}s "
              f"p90={tr[min(int(.9 * len(tr)), len(tr) - 1)]:.1f}s")
    ts, ta = sum(d["committed_syncs"].values()), sum(d["total_syncs"].values())
    print(f"commit rate          {ts}/{ta} = {ts/ta:.1%}")


if __name__ == "__main__":
    main()
