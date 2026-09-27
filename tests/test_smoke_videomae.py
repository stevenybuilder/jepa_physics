"""2-clip CPU smoke test of the VideoMAE-L extraction path (objective axis). Needs MCG-NJU/videomae-large in the HF
cache (~1.4 GB); skipped otherwise."""
import json

import numpy as np
import pytest
import torch

from wm.data import decode, load_table
from wm.extract import (N_POINTS, VIDEOMAE_ID, VM_GRID, VM_TOKENS, disk_mask_videomae, encode_videomae, load_model,
                        merge, preprocess_videomae, run_extraction)

DEVICE = torch.device("cpu")


def _cached():
    try:
        from huggingface_hub import try_to_load_from_cache
        return isinstance(try_to_load_from_cache(VIDEOMAE_ID, "model.safetensors"), str)
    except Exception:
        return False


pytestmark = [pytest.mark.slow, pytest.mark.skipif(not _cached(), reason="videomae-large weights not cached")]


@pytest.fixture(scope="module")
def clips():
    return load_table("direction").iloc[[0, 900]]


@pytest.fixture(scope="module")
def model():
    return load_model("videomae", DEVICE)


def test_points_layout(clips, model):
    frames = decode(clips["video"].iloc[0])
    x = preprocess_videomae([frames])
    assert x.shape == (1, 16, 3, 224, 224)
    points, out = encode_videomae(model, x)
    assert len(points) == N_POINTS
    assert all(p.shape == (1, VM_TOKENS, 1024) for p in points)
    assert torch.equal(points[25], out.last_hidden_state)
    assert torch.allclose(model.layernorm(points[24]), points[25], atol=1e-5)      # point 25 = LN(block 24)
    assert not torch.allclose(points[24], points[25], atol=1e-3)
    # point 0 = patch embedding + fixed sinusoidal position
    pe = model.embeddings.patch_embeddings(x) + model.embeddings.position_embeddings.to(x)
    assert torch.allclose(points[0], pe, atol=1e-5)
    # the hooked block outputs chain: block k+1 applied to point k gives point k+1
    assert torch.allclose(model.encoder.layer[3](points[3]), points[4], atol=1e-4)


def test_resize_matches_hf_processor(clips, monkeypatch):
    """Our torch resize vs VideoMAEImageProcessor (PIL bilinear resize 224 + centre crop 224 = resize on a square
    clip). Reports the difference; only the geometry and normalisation must agree closely."""
    from transformers import VideoMAEImageProcessor
    frames = decode(clips["video"].iloc[0])
    proc = VideoMAEImageProcessor.from_pretrained(VIDEOMAE_ID)
    ref = proc(list(frames), return_tensors="pt")["pixel_values"]                 # [1,16,3,224,224]
    ours = preprocess_videomae([frames])
    diff = (ref - ours).abs()
    print(f"\nmax|ours - HF processor| = {diff.max().item():.3f}, mean = {diff.mean().item():.4f}")
    assert ref.shape == ours.shape
    assert diff.mean().item() < 0.02


def test_diskmask_grid(clips):
    m = disk_mask_videomae(decode(clips["video"].iloc[0]))
    assert m.shape == (8, VM_GRID, VM_GRID) and m.any()


def test_extract_and_merge(clips, tmp_path):
    out_dir = tmp_path / "direction" / "videomae"
    run_extraction(clips, "videomae", out_dir, batch_size=2, device=DEVICE)
    merge(out_dir, clips["id"].tolist())
    index = json.loads((out_dir / "index.json").read_text())
    assert index["model_id"] == VIDEOMAE_ID and index["n_tokens"] == VM_TOKENS and len(index["points"]) == 26
    mean = np.load(out_dir / "meanpool.npy")
    tpool = np.load(out_dir / "timepool.npy")
    dmask = np.load(out_dir / "diskmask.npy")
    assert mean.shape == (2, 26, 1024) and mean.dtype == np.float32
    assert tpool.shape == (2, 26, 8, 1024) and tpool.dtype == np.float16
    assert dmask.shape == (2, 8, VM_GRID, VM_GRID)
    np.testing.assert_allclose(tpool.astype(np.float32).mean(axis=2), mean, rtol=0, atol=2e-2)
