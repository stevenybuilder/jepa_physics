"""Inputs for the Part 2 scripts: one layer of meanpool activations, labels and split roles, in manifest-id order."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wm.data import PROJECT_ROOT, load_table
from wm.splits import load_split

LABEL_COLUMN = {"direction": "theta_degrees", "speed": "speed_mps", "acceleration": "acceleration_mps2"}
KNOT_FOLDS = (0, 1, 2)   # train folds that build PCA + centroids + splines
PROBE_FOLDS = (3, 4)     # train folds that fit the evaluation probe (never knots, never steered)


def default_act_dir(dataset):
    return PROJECT_ROOT / "artifacts" / "activations" / dataset / "vjepa2"


def load_inputs(dataset, layer, variable=None, act_dir=None, table=None, split=None):
    """Returns dict(X [N, D] float32 at `layer`, df (manifest order), y [N] labels of `variable`,
    periodic, role [N] in {"knot", "probe", "test"}, is_train [N]).

    act_dir holds meanpool.npy [N, 26, D] and ids.json written by wm.extract.merge. table (CSV) and split (JSON of
    one dataset's split) override the real manifest and splits/split_v1.json; the tests use them.
    """
    variable = variable or dataset
    act_dir = Path(act_dir) if act_dir else default_act_dir(dataset)
    df = pd.read_csv(table) if table else load_table(dataset)
    s = json.loads(Path(split).read_text()) if split else load_split(dataset)

    ids = json.loads((act_dir / "ids.json").read_text())
    assert ids == [int(i) for i in df["id"]], "activation rows are not in manifest id order"
    X = np.asarray(np.load(act_dir / "meanpool.npy", mmap_mode="r")[:, layer], dtype=np.float32)

    fold = {int(k): v for k, v in s["fold"].items()}
    f = np.array([fold[int(i)] for i in df["id"]])
    role = np.where(f < 0, "test", np.where(np.isin(f, KNOT_FOLDS), "knot", "probe"))
    return {"X": X, "df": df, "y": df[LABEL_COLUMN[variable]].to_numpy(dtype=float),
            "periodic": variable == "direction", "role": role, "is_train": f >= 0}
