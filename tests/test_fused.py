"""The fused decode kernels against the MLX operations they replace: the same
frames, bit for bit. Each run is its own process, since the switches are read
once per process."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SWITCHES = ["D3_NO_FUSED_ATTN", "D3_NO_FUSED_MOE_UP", "D3_NO_FUSED_COMBINE",
            "D3_NO_FUSED_ADDNORM", "D3_NO_FUSED_MIX", "D3_NO_FUSED_EMBED", "D3_NO_FUSED_PAIR",
            "D3_NO_FAST_GEMV", "D3_NO_FUSED_SAMPLER", "D3_NO_FUSED_ROUTER", "D3_NO_FUSED_GDN",
            "D3_NO_COMPILE"]

SCRIPT = """
import sys
import numpy as np
sys.path.insert(0, {tests!r})
import d3
m = d3.Model()
rows = m.build_prompt("<|speaker:0|> [whisper] The quarterly report is due on Thursday, so "
                      "please finish it by noon.")
codes, _ = d3.generate(m, rows, 64, temperature=0.8, top_k=30, top_p=0.9, seed=11)
hidden = d3.prefill(m, rows)
np.savez({out!r}, codes=codes, hidden=hidden[-4:])
"""


def run(tmp_path, name, switches):
    out = tmp_path / f"{name}.npz"
    env = dict(os.environ)
    for s in SWITCHES:
        env.pop(s, None)
    env.update({s: "1" for s in switches})
    script = SCRIPT.format(tests=str(ROOT / "tests"), out=str(out))
    subprocess.run([sys.executable, "-c", script], env=env, check=True, cwd=ROOT)
    return np.load(out)


def test_fused_decode_matches_mlx_ops(tmp_path):
    fused = run(tmp_path, "fused", [])
    ops = run(tmp_path, "ops", SWITCHES)
    assert fused["codes"].shape == ops["codes"].shape
    assert (fused["codes"] == ops["codes"]).all()
    assert (fused["hidden"] == ops["hidden"]).all()
