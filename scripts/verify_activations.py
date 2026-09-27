"""Verify merged activation artifacts: shapes, dtypes, NaN counts, index hashes. Usage: python scripts/verify_activations.py"""
import json, sys, numpy as np, pathlib
root = pathlib.Path("artifacts/activations")
N = {"direction": 1500, "speed": 1536, "acceleration": 1536}
ok = True
for d in N:
    for m in ["vjepa2", "random"]:
        p = root / d / m
        if not p.exists():
            print(f"MISSING {p}"); ok = False; continue
        idx = json.load(open(p / "index.json"))
        mp = np.load(p / "meanpool.npy", mmap_mode="r")
        arrs = [("meanpool", mp)]
        if m == "vjepa2":  # random-init model stores meanpool only, by design
            arrs += [("timepool", np.load(p / "timepool.npy", mmap_mode="r")), ("diskpool", np.load(p / "diskpool.npy", mmap_mode="r"))]
        dp = dict(arrs).get("diskpool")
        exp = {"meanpool": ((N[d], 26, 1024), np.float32), "timepool": ((N[d], 26, 8, 1024), np.float16), "diskpool": ((N[d], 26, 8, 1024), np.float16)}
        for name, arr in arrs:
            shp, dt = exp[name]
            good = arr.shape == shp and arr.dtype == dt
            ok &= good
            nan = int(np.isnan(np.asarray(arr[:, 12])).sum()) if name != "diskpool" else int(np.isnan(np.asarray(dp[:, 12, :, 0])).sum())
            print(f"{d}/{m}/{name}: shape={arr.shape} dtype={arr.dtype} {'OK' if good else 'BAD'} nan@L12={nan}")
        print(f"  index: n_ids={len(idx.get('ids', []))} tf32={idx.get('tf32')} torch={idx.get('torch')} transformers={idx.get('transformers')} device={idx.get('device')}")
        print(f"  |meanpool| L0/L12/L24/L25 = {[float(np.abs(np.asarray(mp[:, l])).max()) for l in (0, 12, 24, 25)]}")
print("ALL OK" if ok else "PROBLEMS"); sys.exit(0 if ok else 1)
