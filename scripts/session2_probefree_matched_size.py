"""QA #402: spline - chord probe-free steering readouts at MATCHED edit size, from cached predictor forecasts only.

The session-2 probe-free readouts (forced choice, graded recovery, 4-way twin identification) at "own size" compare a
spline edit that is larger than the chord's (P22: median 17.4 vs 12.7; D12 per disk token: 88.5 vs 9.2), and QA showed
recovery / forced choice reward a large non-specific edit. This scores paired spline - chord differences where the
two edits have equal size, reusing session2_fourier4_predictor.{forced_choice, recovery_full, twin_identify} and its
reference forecasts / predictor-native heading probe unchanged. No forward pass is run.

  matched pairs (equal size by construction)
    P22_natural         block 22, every context token, delta rescaled per (carrier, target) to ||twin_mp22 - src_mp22||
                        (natural_norm/pred.npz pred_L22[:, 1, {spline, chord}])
    D12_natural_union   block 12, obj = dilate(disk(src) | disk(twin)) tokens, each token unit(delta) x ||h_twin[i] -
                        h_src[i]|| (natural_norm/disk_pred.npz pred[:, {spline, chord}])
    D12_natural_src     same dose on the twin-free set dilate(disk(src)) (disk_sweep box4 src_spline_labelfree_1x vs
                        box2 src_chord_1x)
  size ladders (chord interpolated at the spline's OWN size; forecast linear in edit size between bracketing rungs)
    P22   chord rungs: own (||chord||) and natural (twin norm); spline own = pred_L22[:, 0, 0] at ||spline||
    D12   chord rungs: src set per-token 0.5x / 1x / 2x twin dose (disk_sweep box2, size = dose_mean); spline own =
          fourier4 t1 D12_spline_own (union set, every obj token gets the full spline delta, size = ||spline||).
          Token set (union vs src) and dose shape (uniform vs per-token) differ: a size match, not a like-for-like arm.
  references  fourier4/out/pred.npz rerun src / twin forecasts (the fourier4 score's primary reference)
  CI          run_session2.boot_ci over the 200 carriers of the per-carrier mean over 4 targets (1000 draws, seed 0)

  python scripts/session2_probefree_matched_size.py   -> results/session2_probefree_matched_size.json
"""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent))
from session2_fourier4_predictor import forced_choice, recovery_full, twin_identify  # noqa: E402

READOUTS = ("fc", "rec", "id", "err")
NAMES = {"fc": "forced_choice_frac", "rec": "recovery_full", "id": "twin_id_acc", "err": "heading_err_deg"}


def readouts(E, TW, SRC, head_err=None):
    """E [C, T, 4, D] edited pooled forecasts; TW [C, T, 4, D]; SRC [C, 4, D]. Per-(c, t) readouts dict."""
    E = E.astype(np.float64)
    out = {"fc": forced_choice(E, TW, SRC[:, None]).astype(float), "rec": recovery_full(E, TW, SRC[:, None]),
           "id": twin_identify(E, TW, SRC[:, None]).astype(float)}
    if head_err is not None:
        out["err"] = head_err(E)
    return out


def interp_forecast(sizes, fores, s):
    """Piecewise-linear interpolation of forecasts in edit size, per (c, t).
    sizes [R, C, T] rung sizes; fores [R, C, T, ...] rung forecasts; s [C, T] query size.
    Returns (forecast [C, T, ...], in_bracket [C, T]); outside the rung range the nearest segment is extrapolated."""
    sizes = np.asarray(sizes, float)
    fores = np.asarray(fores, float)
    order = np.argsort(sizes, 0)
    sz = np.take_along_axis(sizes, order, 0)
    fo = np.take_along_axis(fores, order.reshape(order.shape + (1,) * (fores.ndim - 3)), 0)
    R = sz.shape[0]
    seg = np.clip((sz[1:-1] <= s[None]).sum(0) if R > 2 else np.zeros_like(s, int), 0, R - 2).astype(int)
    lo = np.take_along_axis(sz, seg[None], 0)[0]
    hi = np.take_along_axis(sz, seg[None] + 1, 0)[0]
    w = (s - lo) / np.maximum(hi - lo, 1e-12)
    flo = np.take_along_axis(fo, seg.reshape((1,) + seg.shape + (1,) * (fores.ndim - 3)), 0)[0]
    fhi = np.take_along_axis(fo, (seg + 1).reshape((1,) + seg.shape + (1,) * (fores.ndim - 3)), 0)[0]
    wb = w.reshape(w.shape + (1,) * (fores.ndim - 3))
    inb = (s >= sz[0] - 1e-9) & (s <= sz[-1] + 1e-9)
    return (1 - wb) * flo + wb * fhi, inb


def summarize(m, boot_ci, mask=None):
    """m: per-(c, t) readouts -> per-carrier means -> boot_ci; mask [C, T] restricts pairs (carrier mean over kept)."""
    res = {}
    for q, v in m.items():
        if mask is None:
            pc = v.mean(1)
        else:
            k = mask.astype(float)
            pc = np.where(k.sum(1) > 0, (v * k).sum(1) / np.maximum(k.sum(1), 1), np.nan)
        res[NAMES[q]] = boot_ci(pc)
    return res


def paired(ma, mb, boot_ci, mask=None):
    return summarize({q: ma[q] - mb[q] for q in ma}, boot_ci, mask)


def norm_stats(x):
    x = np.asarray(x, float)
    return {"median": float(np.median(x)), "mean": float(x.mean()), "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)), "n_pairs": int(x.size)}


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from run_session2 import angle_of, boot_ci, wrap
    from session2_fourier4_predictor import T1_JOBS
    from wm.data import PROJECT_ROOT, load_table
    from wm.p2_data import load_inputs
    from wm.probes import Standardizer, fit_ridge, predict, targets
    S2 = PROJECT_ROOT / "artifacts" / "session2"
    RES = PROJECT_ROOT / "results"
    files = {"fourier4_pred": S2 / "fourier4/out/pred.npz", "fourier4_plan": S2 / "fourier4/plan.npz",
             "fourier4_forward_info": S2 / "fourier4/out/forward_info.json",
             "natural_norm_pred": S2 / "natural_norm/pred.npz", "natural_norm_disk_pred": S2 / "natural_norm/disk_pred.npz",
             "natural_norm_deltas": S2 / "natural_norm/deltas.npz",
             "disk_sweep_box2": S2 / "disk_sweep/shards/box2/sweep_pred.npz",
             "disk_sweep_box4": S2 / "disk_sweep/shards/box4/sweep_pred.npz",
             "native_pred_pooled_all": S2 / "native/pred_pooled_all.npy",
             "native_readout_json": RES / "session2_predictor_native_readout.json"}
    z, pz = np.load(files["fourier4_pred"]), np.load(files["fourier4_plan"])
    nn, dk, de = np.load(files["natural_norm_pred"]), np.load(files["natural_norm_disk_pred"]), np.load(files["natural_norm_deltas"])
    b2, b4 = np.load(files["disk_sweep_box2"]), np.load(files["disk_sweep_box4"])
    C = len(z["idx"])
    for d in (pz, nn, dk, de, b2, b4):
        assert (d["idx"][:C] == z["idx"]).all()
    rows, tg = pz["carrier_rows"][:C], pz["targets"][:C]
    SRC, TW = z["src_pred_pooled"].astype(np.float64), z["twin_pred_pooled"].astype(np.float64)
    # predictor-native heading probe: identical to session2_fourier4_predictor.score
    nat_json = json.loads(files["native_readout_json"].read_text())
    df = load_table("direction")
    Y = targets(df, "direction")[0]
    probe = load_inputs("direction", 0)["role"] == "probe"
    Zall = np.load(files["native_pred_pooled_all"]).astype(np.float64)
    st = Standardizer().fit(Zall.mean(1)[probe])
    W, b = fit_ridge(st.transform(Zall.mean(1)[probe]), Y[probe],
                     nat_json["direction"]["predictor_output_probe_stepmean"]["alpha"])
    readP = lambda X: predict(st.transform(X.reshape(-1, X.shape[-1])), W, b).reshape(X.shape[:-1] + (-1,))  # noqa
    head_err = lambda E: wrap(angle_of(readP(E.mean(2))) - tg)  # noqa
    M = lambda E: readouts(E, TW, SRC, head_err)  # noqa
    J = {j[0]: i for i, j in enumerate(T1_JOBS)}
    arms_b2, arms_b4 = list(b2["arms"]), list(b4["arms"])
    nat22, nat12 = de["natural_norm_L22"][:C].astype(float), de["natural_norm_L12"][:C].astype(float)
    own22, own12 = de["own_norm_L22"][:C].astype(float), de["own_norm_L12"][:C].astype(float)   # [C, arm, T]

    out = {"question": "QA #402: does the spline still lead the chord on the probe-free readouts once edit size is "
                       "matched?", "n_carriers": int(C), "n_targets": int(tg.shape[1]), "matched": {}, "ladder": {}}
    # ---------- matched (equal size by construction)
    pairs = {
        "P22_natural": (nn["pred_L22"][:C, 1, 0], nn["pred_L22"][:C, 1, 1],
                        {"spline": norm_stats(nat22), "chord": norm_stats(nat22),
                         "unit": "L2 of the pooled delta added to every context token at block 22 (= ||twin_mp22 - src_mp22||)"}),
        "D12_natural_union": (dk["pred"][:C, 0], dk["pred"][:C, 1],
                              {"spline_dose_mean_per_token": norm_stats(b2["dose_mean"][:C, arms_b2.index("union_chord_1x")]),
                               "chord_dose_mean_per_token": norm_stats(b2["dose_mean"][:C, arms_b2.index("union_chord_1x")]),
                               "n_tokens": norm_stats(dk["ntok"][:C]),
                               "unit": "per obj token ||h_twin[i] - h_src[i]|| at block 12 (identical for both arms; "
                                       "mean over the union set; values from the disk_sweep union_chord_1x rerun of the same set)"}),
        "D12_natural_src": (b4["pred"][:C, arms_b4.index("src_spline_labelfree_1x")],
                            b2["pred"][:C, arms_b2.index("src_chord_1x")],
                            {"spline_dose_mean_per_token": norm_stats(b4["dose_mean"][:C, arms_b4.index("src_spline_labelfree_1x")]),
                             "chord_dose_mean_per_token": norm_stats(b2["dose_mean"][:C, arms_b2.index("src_chord_1x")]),
                             "n_tokens": norm_stats(b2["ntok"][:C, arms_b2.index("src_chord_1x")]),
                             "unit": "per token ||h_twin[i] - h_src[i]|| on dilate(disk(src)) at block 12"}),
    }
    par = {}
    for name, (Es, Ec, norms) in pairs.items():
        ms, mc = M(Es), M(Ec)
        out["matched"][name] = {"spline": summarize(ms, boot_ci), "chord": summarize(mc, boot_ci),
                                "spline_minus_chord": paired(ms, mc, boot_ci), "edit_size": norms}
    if "D12_natural_src" in pairs:
        dm_s = b4["dose_mean"][:C, arms_b4.index("src_spline_labelfree_1x")]
        dm_c = b2["dose_mean"][:C, arms_b2.index("src_chord_1x")]
        par["D12_natural_src_dose_mean_rel_maxabs_spline_vs_chord"] = float(np.abs(dm_s - dm_c).max() / np.abs(dm_c).max())
    # parity with the published fourier4 table (same reference, same readouts)
    f4 = json.loads((RES / "session2_fourier4_predictor.json").read_text())["task1"]["table"]
    for site, key in (("P22", "P22_natural"), ("D12", "D12_natural_union")):
        for arm in ("spline", "chord"):
            for q in ("forced_choice_frac", "recovery_full", "twin_id_acc"):
                par[f"{key}_{arm}_{q}_minus_fourier4_table"] = out["matched"][key][arm][q]["mean"] - f4[site]["natural"][arm][q]["mean"]
            par[f"{key}_{arm}_heading_err_minus_fourier4_table"] = (out["matched"][key][arm]["heading_err_deg"]["mean"]
                                                                   - f4[site]["natural"][arm]["err_to_target"]["mean"])
    out["parity"] = par

    # ---------- ladders: chord at the spline's own size
    # P22: rungs chord own / chord natural
    s22 = own22[:, 0]
    sizes = np.stack([own22[:, 1], nat22])
    fores = np.stack([nn["pred_L22"][:C, 0, 1], nn["pred_L22"][:C, 1, 1]]).astype(np.float64)
    Ec22, inb22 = interp_forecast(sizes, fores, s22)
    ms_own22, mc22 = M(nn["pred_L22"][:C, 0, 0]), M(Ec22)
    out["ladder"]["P22"] = {
        "rungs": {"chord_own": norm_stats(own22[:, 1]), "chord_natural": norm_stats(nat22)},
        "spline_own_size": norm_stats(s22), "spline_over_chord_own_ratio": norm_stats(own22[:, 0] / own22[:, 1]),
        "frac_pairs_in_bracket": float(inb22.mean()),
        "spline_own": summarize(ms_own22, boot_ci), "chord_interp_at_spline_size": summarize(mc22, boot_ci),
        "spline_minus_chord_all_pairs": paired(ms_own22, mc22, boot_ci),
        "spline_minus_chord_in_bracket_pairs": paired(ms_own22, mc22, boot_ci, inb22)}
    # D12: rungs src chord 0.5x / 1x / 2x (size = mean per-token dose); spline own = union set, full delta per token
    names = ["src_chord_0.5x", "src_chord_1x", "src_chord_2x"]
    sizes = np.stack([b2["dose_mean"][:C, arms_b2.index(a)] for a in names]).astype(float)
    fores = np.stack([b2["pred"][:C, arms_b2.index(a)] for a in names]).astype(np.float64)
    s12 = own12[:, 0]
    Ec12, inb12 = interp_forecast(sizes, fores, s12)
    ms_own12, mc12 = M(z["t1"][:, J["D12_spline_own"]]), M(Ec12)
    out["ladder"]["D12"] = {
        "rungs": {a: norm_stats(sizes[i]) for i, a in enumerate(names)},
        "spline_own_size": norm_stats(s12), "chord_own_size": norm_stats(own12[:, 1]),
        "frac_pairs_in_bracket": float(inb12.mean()),
        "spline_own": summarize(ms_own12, boot_ci), "chord_interp_at_spline_size": summarize(mc12, boot_ci),
        "spline_minus_chord_all_pairs": paired(ms_own12, mc12, boot_ci),
        "spline_minus_chord_in_bracket_pairs": paired(ms_own12, mc12, boot_ci, inb12),
        "caveat": "token sets differ (spline own: union obj set, uniform full delta; chord rungs: src set, per-token twin "
                  "dose): matched on per-token edit size only"}
    out["definitions"] = {
        "forced_choice_frac": "fraction of (carrier, target) pairs whose edited pooled forecast (4 steps x 1024) is closer "
                              "in L2 to the steered twin's forecast than to the unedited forecast "
                              "(session2_fourier4_predictor.forced_choice)",
        "recovery_full": "<e - src, tw - src> / ||tw - src||^2 over the pooled forecast (recovery_full); fc == rec > 0.5",
        "twin_id_acc": "4-way: the edit's forecast change has its highest cosine with its own target's twin change "
                       "among the carrier's 4 twins (twin_identify); chance 0.25",
        "heading_err_deg": "wrapped angle between the predictor-native direction probe's read of the step-mean edited "
                           "forecast and the target (as session2_fourier4_predictor 'err_to_target'); lower is better",
        "reference_forecasts": "fourier4/out/pred.npz src_pred_pooled / twin_pred_pooled (fourier4 score's primary reference)",
        "spline_minus_chord": "per (carrier, target) difference, averaged over the carrier's targets, boot_ci over carriers "
                              "(1000 draws, seed 0); positive = spline higher (for heading_err, positive = spline worse)",
        "matched_size": "natural arms: both edits rescaled to the twin's own change (P22 pooled norm; D12 per-token norm), "
                        "so sizes are equal by construction",
        "ladder_interpolation": "chord forecast linearly interpolated in edit size between the bracketing chord rungs at "
                                "the spline's own size per (carrier, target); outside the rungs the nearest segment is "
                                "extrapolated (in_bracket subset reported separately). Assumes the forecast is locally "
                                "linear in edit size.",
        "n": "number of carriers entering the bootstrap (per-carrier mean over its 4 targets; in-bracket: carriers with >= 1 "
             "in-bracket pair)"}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip())
    out["provenance"] = {"inputs_sha256": {k: {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256_file(p)}
                                           for k, p in files.items()},
                         "script": str(HERE.relative_to(PROJECT_ROOT)), "script_sha256": sha256_file(HERE),
                         "commit": commit, "dirty": dirty, "seeds": {"bootstrap": 0}, "n_boot": 1000,
                         "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                         "gpu": None, "stage": "probefree_matched_size (CPU, cached forecasts only)"}
    (RES / "session2_probefree_matched_size.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    res = main()
    f = lambda d: f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] n={d['n']}"  # noqa
    for k, v in res["matched"].items():
        print(k, {q: f(v["spline_minus_chord"][q]) for q in v["spline_minus_chord"]})
    for k, v in res["ladder"].items():
        print("ladder", k, "in_bracket", round(v["frac_pairs_in_bracket"], 3),
              {q: f(v["spline_minus_chord_all_pairs"][q]) for q in v["spline_minus_chord_all_pairs"]})
    print(json.dumps(res["parity"], indent=0))
