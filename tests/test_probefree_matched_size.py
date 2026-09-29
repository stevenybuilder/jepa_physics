import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from session2_probefree_matched_size import interp_forecast, paired, readouts, summarize  # noqa: E402
from run_session2 import boot_ci  # noqa: E402


def test_interp_forecast_linear_and_bracket():
    C, T = 3, 4
    rng = np.random.default_rng(0)
    base, slope = rng.normal(size=(C, T, 4, 5)), rng.normal(size=(C, T, 4, 5))
    sizes = np.stack([np.full((C, T), 1.0), np.full((C, T), 2.0), np.full((C, T), 4.0)])
    fores = np.stack([base + slope * s[..., None, None] for s in sizes])
    s = np.array([[1.5, 3.0, 4.0, 5.0]] * C)
    f, inb = interp_forecast(sizes, fores, s)
    assert np.allclose(f, base + slope * s[..., None, None])        # exact on a linear response, incl. extrapolation
    assert inb.tolist() == [[True, True, True, False]] * C
    f2, _ = interp_forecast(sizes[::-1], fores[::-1], s)             # rung order does not matter
    assert np.allclose(f, f2)


def test_readouts_twin_and_src_and_paired():
    rng = np.random.default_rng(1)
    C, T = 20, 4
    SRC = rng.normal(size=(C, 4, 6))
    TW = SRC[:, None] + rng.normal(size=(C, T, 4, 6))
    m_tw = readouts(TW, TW, SRC)
    assert m_tw["fc"].all() and np.allclose(m_tw["rec"], 1) and m_tw["id"].all()
    m_src = readouts(np.repeat(SRC[:, None], T, 1) + 1e-9, TW, SRC)
    assert not m_src["fc"].any() and np.allclose(m_src["rec"], 0, atol=1e-6)
    d = paired(m_tw, m_src, boot_ci)
    assert d["forced_choice_frac"]["mean"] == 1.0 and d["forced_choice_frac"]["n"] == C
    mask = np.zeros((C, T), bool)
    mask[:5, 0] = True
    assert summarize(m_tw, boot_ci, mask)["recovery_full"]["n"] == 5
