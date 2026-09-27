"""8-clip CPU smoke test of the extraction path (~2-3 min on the Intel Mac)."""
import json
import time

import numpy as np
import pytest
import torch
from transformers import AutoVideoProcessor

from wm.data import decode, load_table
from wm.extract import (IMAGENET_MEAN, IMAGENET_STD, MODEL_ID, N_POINTS, N_TOKENS, encode,
                        load_model, merge, preprocess, run_extraction)

DEVICE = torch.device("cpu")


@pytest.fixture(scope="module")
def speed8():
    df = load_table("speed")
    return df.iloc[::192].head(8)          # 8 clips spread over the 64 speeds


@pytest.fixture(scope="module")
def model():
    return load_model("vjepa2", DEVICE)


@pytest.fixture(scope="module")
def one_clip(speed8, model):
    frames = decode(speed8["video"].iloc[0])
    t0 = time.time()
    points, out = encode(model, preprocess([frames]))
    print(f"\nsingle-clip forward: {time.time() - t0:.1f} s")
    return frames, points, out


@pytest.fixture(scope="module")
def extracted(speed8, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("act") / "speed" / "vjepa2"
    t0 = time.time()
    run_extraction(speed8, "vjepa2", out_dir, batch_size=8, device=DEVICE)
    print(f"\n8-clip extraction: {time.time() - t0:.1f} s")
    merge(out_dir, speed8["id"].tolist())
    return out_dir


def test_hidden_states_and_hook(one_clip, model):
    _, points, out = one_clip
    assert len(out.hidden_states) == 25
    assert len(points) == N_POINTS
    for p in points:
        assert p.shape == (1, N_TOKENS, 1024)
    hs = out.hidden_states
    assert torch.allclose(hs[24], out.last_hidden_state)
    hook = points[24]
    assert not torch.allclose(hook, hs[24], atol=1e-3)              # pre-LN != post-LN
    assert torch.allclose(model.encoder.layernorm(hook), hs[24], atol=1e-5)
    assert torch.equal(points[23], hs[23]) and torch.equal(points[25], hs[24])


def test_processor_deviation(one_clip, monkeypatch):
    frames, _, _ = one_clip
    # transformers 4.56.2's resize path calls torch.compiler.is_compiling, added after torch 2.2.
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False, raising=False)
    proc = AutoVideoProcessor.from_pretrained(MODEL_ID)
    assert tuple(proc.image_mean) == IMAGENET_MEAN and tuple(proc.image_std) == IMAGENET_STD
    ours = preprocess([frames])
    no_resize = proc(videos=[frames], do_resize=False, do_center_crop=False,
                     return_tensors="pt")["pixel_values_videos"]
    default = proc(videos=[frames], return_tensors="pt")["pixel_values_videos"]
    print(f"\nmax|ours - processor(no resize/crop)| = {(no_resize - ours).abs().max().item():.2e}; "
          f"max|ours - processor(default)| = {(default - ours).abs().max().item():.2f}")
    assert no_resize.shape == ours.shape == (1, 16, 3, 256, 256)
    assert (no_resize - ours).abs().max().item() < 1e-5
    assert default.shape == ours.shape
    assert (default - ours).abs().max().item() > 0.1          # resize 292 + crop 256 is not a no-op


def test_pools(extracted, speed8):
    index = json.loads((extracted / "index.json").read_text())
    assert index["ids"] == speed8["id"].tolist()
    assert len(index["seconds_per_clip"]) == 8
    mean = np.load(extracted / "meanpool.npy")
    tpool = np.load(extracted / "timepool.npy")
    dpool = np.load(extracted / "diskpool.npy")
    dmask = np.load(extracted / "diskmask.npy")
    assert mean.shape == (8, N_POINTS, 1024) and mean.dtype == np.float32
    assert tpool.shape == dpool.shape == (8, N_POINTS, 8, 1024) and tpool.dtype == np.float16
    assert dmask.shape == (8, 8, 16, 16) and dmask.dtype == bool
    # mean over all tokens == mean over time of the per-step means (fp16 storage tolerance)
    np.testing.assert_allclose(tpool.astype(np.float32).mean(axis=2), mean, rtol=2e-3, atol=2e-3)
    assert not np.isnan(dpool).any()
    assert dmask.any(axis=(2, 3)).all()
    # the disk mean is a different vector from the whole-frame mean
    assert not np.allclose(dpool.astype(np.float32), tpool.astype(np.float32), atol=1e-2)


def test_resume_skips_existing(extracted, speed8, capsys):
    run_extraction(speed8, "vjepa2", extracted, batch_size=8, device=DEVICE)
    assert "chunk 0: exists, skipping" in capsys.readouterr().out
