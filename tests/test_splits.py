import pytest

from wm.data import DATASETS, load_table
from wm.splits import N_FOLDS, load_split, validate_split


@pytest.mark.parametrize("dataset", DATASETS)
def test_split(dataset):
    assert validate_split(dataset)
    s = load_split(dataset)
    df = load_table(dataset)
    c = s["counts"]
    assert c["n_train"] + c["n_test"] == c["n_clips"] == len(df)
    assert sum(c["fold_sizes"].values()) == c["n_train"]
    assert abs(c["n_test"] / c["n_clips"] - 0.2) < 0.01

    label_of = dict(zip(df["id"], df["label"]))
    values = set(df["label"])
    assert {label_of[i] for i in s["train_ids"]} == values
    assert {label_of[i] for i in s["test_ids"]} == values

    fold = {int(k): v for k, v in s["fold"].items()}
    assert {i for i in s["train_ids"]} == {i for i, f in fold.items() if f >= 0}
    assert {fold[i] for i in s["train_ids"]} == set(range(N_FOLDS))


def test_validate_split_catches_leak():
    s = load_split("speed")
    bad = dict(s, test_ids=s["test_ids"] + [s["train_ids"][0]])
    with pytest.raises(AssertionError):
        validate_split("speed", bad)


def test_split_env_override(monkeypatch):
    import importlib

    import wm.splits as splits
    from wm.data import PROJECT_ROOT
    monkeypatch.delenv("WM_SPLIT_PATH", raising=False)
    monkeypatch.delenv("WM_TEST_SIZE", raising=False)
    importlib.reload(splits)
    assert splits.SPLIT_PATH == PROJECT_ROOT / "splits" / "split_v1.json" and splits.TEST_SIZE == 0.2
    monkeypatch.setenv("WM_SPLIT_PATH", "splits/split_paper70.json")
    monkeypatch.setenv("WM_TEST_SIZE", "0.3")
    importlib.reload(splits)
    assert splits.SPLIT_PATH == PROJECT_ROOT / "splits" / "split_paper70.json" and splits.TEST_SIZE == 0.3
    monkeypatch.delenv("WM_SPLIT_PATH")
    monkeypatch.delenv("WM_TEST_SIZE")
    importlib.reload(splits)
