"""The one train/test split per dataset (spec section 3), stored in splits/split_v1.json.

Protocol: clips with byte-identical decoded frames are merged into one unit; units are split
80/20 at random (seed 0), stratified by label value; 5 stratified folds (seed 0) inside train.
Test clips get fold = -1.

Env overrides (sensitivity runs only; defaults unchanged): WM_SPLIT_PATH (file, relative to the project root unless
absolute) and WM_TEST_SIZE (test fraction used by make_split).
"""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from wm.data import PROJECT_ROOT, load_table

SPLIT_PATH = PROJECT_ROOT / os.environ.get("WM_SPLIT_PATH", "splits/split_v1.json")  # absolute env path wins
SEED = 0
TEST_SIZE = float(os.environ.get("WM_TEST_SIZE", 0.2))
N_FOLDS = 5


def make_split(df, hashes):
    """df: load_table output; hashes: {id: frame_hash}. Returns the split dict for one dataset."""
    ids = df["id"].to_numpy()
    label_of = dict(zip(ids, df["label"]))

    # Units = groups of identical clips. A unit takes the label of its clips (they must agree).
    units = {}
    for i in ids:
        units.setdefault(hashes[i], []).append(int(i))
    unit_ids = list(units.values())
    unit_labels = []
    for members in unit_ids:
        labels = {label_of[i] for i in members}
        assert len(labels) == 1, f"identical clips with different labels: {members}"
        unit_labels.append(labels.pop())
    unit_labels = np.array(unit_labels)
    _, unit_y = np.unique(unit_labels, return_inverse=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_u, test_u = next(sss.split(np.zeros(len(unit_ids)), unit_y))

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = {int(i): -1 for i in ids}
    for k, (_, val_idx) in enumerate(skf.split(np.zeros(len(train_u)), unit_y[train_u])):
        for u in train_u[val_idx]:
            for i in unit_ids[u]:
                fold[i] = k

    train_ids = sorted(i for u in train_u for i in unit_ids[u])
    test_ids = sorted(i for u in test_u for i in unit_ids[u])
    return {
        "seed": SEED,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "fold": {str(i): fold[i] for i in sorted(fold)},
        "frame_hash": {str(i): hashes[i] for i in sorted(hashes)},
        "counts": {
            "n_clips": len(ids),
            "n_units": len(unit_ids),
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_train_units": len(train_u),
            "n_test_units": len(test_u),
            "fold_sizes": {str(k): v for k, v in sorted(Counter(fold[i] for i in train_ids).items())},
            "n_label_values": int(len(np.unique(unit_labels))),
        },
    }


def load_split(dataset, path=None):
    return json.loads((SPLIT_PATH if path is None else Path(path)).read_text())[dataset]


def validate_split(dataset, split=None, df=None):
    """Raise AssertionError if the split breaks any rule of the protocol (df: a load_table(root=...) override)."""
    s = split if split is not None else load_split(dataset)
    df = load_table(dataset) if df is None else df
    train, test = set(s["train_ids"]), set(s["test_ids"])
    manifest = set(int(i) for i in df["id"])

    both = train & test
    if both:
        raise AssertionError(f"{dataset}: {len(both)} ids in both train and test")
    if train | test != manifest:
        raise AssertionError(f"{dataset}: split ids do not cover the manifest exactly")
    if len(s["train_ids"]) != len(train) or len(s["test_ids"]) != len(test):
        raise AssertionError(f"{dataset}: repeated ids")

    label_of = dict(zip(df["id"], df["label"]))
    all_values = set(df["label"])
    for name, part in (("train", train), ("test", test)):
        missing = all_values - {label_of[i] for i in part}
        if missing:
            raise AssertionError(f"{dataset}: {len(missing)} label values missing from {name}")

    side = {}
    for i in manifest:
        h = s["frame_hash"][str(i)]
        where = "train" if i in train else "test"
        if side.setdefault(h, where) != where:
            raise AssertionError(f"{dataset}: identical clips (hash {h[:12]}) on both sides")

    fold = {int(k): v for k, v in s["fold"].items()}
    if set(fold) != manifest:
        raise AssertionError(f"{dataset}: fold map does not cover the manifest")
    if any(fold[i] == -1 for i in train) or any(fold[i] != -1 for i in test):
        raise AssertionError(f"{dataset}: fold -1 must mean exactly the test clips")
    if {fold[i] for i in train} != set(range(N_FOLDS)):
        raise AssertionError(f"{dataset}: train folds are not 0..{N_FOLDS - 1}")
    return True
