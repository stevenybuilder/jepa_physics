"""Refresh a stored context-run result of run_part2.py without re-steering: the dual-floor verdict (context_verdict),
the sha256 of the context table/split, and the gap and waypoint figures, all from the JSON's stored summary/gaps.

  python scripts/replot_part2_context.py results/p2_steer_direction_direction_L12_contiguous_ctx-hard.json \
      --table results/ctx_hard_inputs/hard_table.csv --split results/ctx_hard_inputs/split_hard_direction.json
"""
import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_part2", ROOT / "scripts" / "run_part2.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)


def refresh(path, figures_dir, table=None, split=None):
    out = json.loads(Path(path).read_text())
    periodic = out["variable"] == "direction"
    c = out["context"]
    assert c is not None, "not a context run"
    for key, f in (("table", table), ("split", split)):
        if f:
            sha = p2._sha256(f)
            if c.get(f"steered_{key}_sha256"):
                assert c[f"steered_{key}_sha256"] == sha, f"{key} differs from the one the run recorded"
            c[f"steered_{key}_sha256"], c[f"steered_{key}_repo_copy"] = sha, str(Path(f))
    old = out["verdict"]
    out["verdict"] = p2.context_verdict(out, periodic)
    assert out["verdict"]["call"] == old["call"] and out["verdict"]["text"] == old["text"], "mean-floor verdict drifted"
    out["summary_notes"][0] = "verdict: " + out["verdict"]["text"]
    out["regenerated"] = {"by": "scripts/replot_part2_context.py", "utc": datetime.now(timezone.utc).isoformat(
        timespec="seconds"), "what": "verdict margin_floors + margin_note, context sha256s, gap and waypoint figures; "
        "no steering rerun (rows, summary and gaps are the original run's)"}
    Path(path).write_text(json.dumps(out, indent=1))
    tag = Path(path).stem.removeprefix("p2_steer_")
    main = [a for a in out["arms"] if a != "goodfire_linear"]
    verdict = "verdict: " + out["verdict"]["text"]
    figs = [Path(figures_dir) / f"fig4_gap_vs_shift_{tag}.png", Path(figures_dir) / f"fig4_waypoint_readout_{tag}.png"]
    p2.plot_gap(out["summary"], figs[0], tag, periodic, main, verdict)
    p2.plot_waypoints(out["waypoint_readout"], figs[1], tag, periodic, main, out["behaviour_floor"]["mean"], verdict,
                      context=True)
    return out, figs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+")
    ap.add_argument("--table")
    ap.add_argument("--split")
    ap.add_argument("--figures-dir", default=str(ROOT / "figures"))
    a = ap.parse_args()
    for r in a.results:
        o, figs = refresh(r, a.figures_dir, a.table, a.split)
        print(r, o["verdict"]["margin_floors"]["call_under_mean_floor"],
              o["verdict"]["margin_floors"]["call_under_median_floor"], *figs)
