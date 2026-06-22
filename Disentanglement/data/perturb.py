"""Speaker perturbation for invariance training (NANSY-style).

Changes the SPEAKER (pitch F0 + formants/vocal-tract) while preserving the
linguistic CONTENT and the TIMING (no time-stretch), so frame t of the original
aligns with frame t of the perturbed signal.  Used to enforce z_L invariance:
z_L(x) ~= z_L(perturb(x))  =>  z_L carries content, not speaker.
"""
from __future__ import annotations

import numpy as np

try:
    import pyworld as pw
    _FP = pw.default_frame_period          # 5.0 ms
except Exception:                          # pragma: no cover
    pw = None


def perturb_speaker(wav, sr: int = 16_000,
                    f0_range=(0.7, 1.5), formant_range=(0.85, 1.3),
                    f0_scale=None, formant_scale=None):
    """Return a same-length waveform with speaker changed, content/timing kept.

    f0_scale>1 raises pitch; formant_scale>1 shifts formants up (shorter tract).
    Scales are drawn uniformly from the ranges if not given.
    """
    if pw is None:
        raise RuntimeError("pyworld not installed — needed for speaker perturbation.")
    n = int(len(wav))
    if n < 2048:                           # too short to analyse
        return np.asarray(wav, dtype=np.float32).copy()
    x = np.asarray(wav, dtype=np.float64)
    if f0_scale is None:
        f0_scale = float(np.random.uniform(*f0_range))
    if formant_scale is None:
        formant_scale = float(np.random.uniform(*formant_range))

    # WORLD analysis (dio+stonemask = fast F0)
    f0, t = pw.dio(x, sr, frame_period=_FP)
    f0 = pw.stonemask(x, f0, t, sr)
    sp = pw.cheaptrick(x, f0, t, sr)       # spectral envelope (frames, nbin)
    ap = pw.d4c(x, f0, t, sr)              # aperiodicity

    # --- F0 perturbation ---
    f0_p = f0 * f0_scale

    # --- formant perturbation: warp the frequency axis of the envelope ---
    # new[:, k] = sp[:, k / formant_scale]   (linear interp, vectorised over frames)
    nbin = sp.shape[1]
    src = np.clip(np.arange(nbin) / formant_scale, 0, nbin - 1)
    lo  = np.floor(src).astype(np.int64)
    hi  = np.clip(lo + 1, 0, nbin - 1)
    frac = (src - lo)[None, :]
    sp_p = sp[:, lo] * (1.0 - frac) + sp[:, hi] * frac
    sp_p = np.ascontiguousarray(np.maximum(sp_p, 1e-16))

    y = pw.synthesize(f0_p, sp_p, ap, sr, _FP)
    if len(y) >= n:
        y = y[:n]
    else:
        y = np.pad(y, (0, n - len(y)))
    return y.astype(np.float32)
