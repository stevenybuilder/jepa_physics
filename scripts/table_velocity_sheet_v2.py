"""Markdown tables from results/p5_velocity_sheet_v2.json (or an acceleration file): python scripts/table_velocity_sheet_v2.py FILE"""
import json
import sys

d = json.load(open(sys.argv[1]))["results"]
unit = "m/s" if "velocity" in sys.argv[1] else "m/s^2"


def f(s, p=2):
    return f"{s['mean']:.{p}f} [{s['ci95'][0]:.{p}f}, {s['ci95'][1]:.{p}f}]"


def row(summ, gaps, arm, cols):
    s = summ[arm]
    out = [arm] + [f(s[c], p) if c in s else "-" for c, p in cols]
    g = gaps.get(arm, {}).get("err_dir")
    gs = gaps.get(arm, {}).get("err_spd")
    out += [f(g) if g else "-", f(gs, 3) if gs else "-"]
    return "| " + " | ".join(out) + " |"


for L in sorted(d, key=int):
    for de in ("block2", "block3", "cross"):
        if de not in d[L]:
            continue
        D = d[L][de]
        rf = D["reader_floor"]["real"]["summary"]["real_target_clips"]
        print(f"\n### point {L}, {de}: joint steer to held-out cells (matched norm = every arm at the sheet's per-clip norm)")
        print(f"reader floor on real target-cell test clips: dir {f(rf['err_dir'])} deg, speed {f(rf['err_spd'], 3)}; "
              f"MLP dir {f(rf['err_dir_mlp'])}, speed {f(rf['err_spd_mlp'], 3)}; n={rf['err_dir']['n']}")
        for cond in ("matched", "own"):
            node = D["joint"][cond]
            cols = [("err_dir", 2), ("err_spd", 3), ("err_dir_mlp", 2), ("err_spd_mlp", 3), ("nearest_real_R", 3),
                    ("leak_pos", 3), ("edit_norm", 1)]
            print(f"\n{cond} norm. key: results.{L}.{de}.joint.{cond}.summary.<arm>.<metric>; gaps: .gaps_sheet_minus.<arm>")
            print("| arm | dir err (deg) | speed err | MLP dir | MLP speed | nearest-real R | start-pos leak (m) | |edit| | sheet−arm dir | sheet−arm speed |")
            print("|" + "---|" * 10)
            for arm in node["summary"]:
                if arm == "unedited" and cond == "matched":
                    continue
                print(row(node["summary"], node.get("gaps_sheet_minus", {}), arm, cols))
            print("per-seed sheet−seq_local_dir_then_speed dir gap:", [round(x, 2) for x in node.get("per_seed_gap_err_dir_vs_seq_local_dir_then_speed", [])])
        jp = D.get("joint_path", {}).get("own", {}).get("summary")
        if jp:
            print("path (own): sheet", {k: round(v["mean"], 3) for k, v in jp["sheet"].items()},
                  "| chord_sheet", {k: round(v["mean"], 3) for k, v in jp["chord_sheet"].items()})
        for task in ("direction", "speed"):
            if task not in D:
                continue
            node = D[task]["matched"]
            cols = [("err_dir", 2), ("err_spd", 3), ("leak_spd", 3), ("leak_dir", 2), ("leak_pos", 3), ("nearest_real_R", 3),
                    ("err_dir_mlp", 2), ("err_spd_mlp", 3)]
            print(f"\n{task}-only steer, matched norm (key results.{L}.{de}.{task}.matched)")
            print("| arm | dir err | speed err | speed leak | dir leak | pos leak | R | MLP dir | MLP speed | sheet−arm dir | sheet−arm speed |")
            print("|" + "---|" * 11)
            for arm in node["summary"]:
                print(row(node["summary"], node.get("gaps_sheet_minus", {}), arm, cols))
            gl = node.get("gaps_sheet_minus", {})
            key = "leak_spd" if task == "direction" else "leak_dir"
            print(f"sheet−arm {key}:", {a: f(g[key], 3) for a, g in gl.items() if key in g})
            own = D[task]["own"]["summary"]
            print("own-norm err:", {a: round(own[a]["err_dir" if task == "direction" else "err_spd"]["mean"], 3) for a in own},
                  "own-norm |edit|:", {a: round(own[a]["edit_norm"]["mean"], 2) for a in own})
        for task in ("dose_direction", "dose_speed"):
            if task not in D:
                continue
            print(f"\n{task} (key results.{L}.{de}.{task}.x<dose>.summary.<arm>)")
            leak = "leak_spd" if task == "dose_direction" else "leak_dir"
            print(f"| dose | arm | achieved fraction | {leak} (ridge) | {leak} (MLP) | pos leak |")
            print("|---|---|---|---|---|---|")
            for dz in sorted(D[task], key=lambda k: float(k[1:])):
                for arm, s in D[task][dz]["summary"].items():
                    print(f"| {dz} | {arm} | {f(s['achieved_frac'])} | {f(s[leak], 3)} | {f(s[leak + '_mlp'], 3)} | {f(s['leak_pos'], 3)} |")
    if "extrapolate" in d[L]:
        print(f"\n### point {L} extrapolation (speed-only; key results.{L}.extrapolate.extrapolate.v<target>)")
        print("| target | arm | ridge read | MLP read | |read−target| | dir leak | |edit| |")
        print("|---|---|---|---|---|---|---|")
        for tv, node in d[L]["extrapolate"]["extrapolate"].items():
            for arm, s in node["summary"].items():
                print(f"| {tv} | {arm} | {f(s['read_spd'])} | {f(s['read_spd_mlp'])} | {f(s['err_spd'], 3)} | {f(s['leak_dir'])} | {s['edit_norm']['mean']:.1f} |")
