"""
One-shot script: captures the PRE-CHANGE analyze_video_blobtrack output
(stride=3) on the deterministic tiny_video fixture, saved as a regression
golden for Phase 1. Run once, before modifying keyseer/blobtrack.py:

    python -m tests.fixtures._generate_golden

Do not re-run after blobtrack.py changes -- that would defeat the point
of a pre-change regression baseline.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tests.conftest import _write_noisy_video
from keyseer.blobtrack import analyze_video_blobtrack

OUT = Path(__file__).resolve().parent / "golden_blobtrack_stride3.npz"


def main():
    video_path = str(Path(__file__).resolve().parent / "_golden_tmp.mp4")
    _write_noisy_video(video_path, n_frames=120, event=(20, 100), fps=10, seed=0)
    r = analyze_video_blobtrack(video_path, resize_to=(180, 320), stride=3)
    np.savez(OUT, scores=r["scores"], n_active=r["n_active"], n_frames=r["n_frames"])
    Path(video_path).unlink(missing_ok=True)
    print(f"saved golden to {OUT}: n_frames={r['n_frames']}, "
         f"scores.shape={r['scores'].shape}, sum(scores)={r['scores'].sum():.6f}")


if __name__ == "__main__":
    main()
