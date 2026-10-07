"""NeoTex belt QRS detector — self-contained, 100 Hz, causal, no external ECG library.
Pipeline: bandpass -> derivative -> square -> moving integration -> adaptive threshold w/ refractory + search-back.
"""
import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi

def design(fs=100, lo=8, hi=30, order=2):
    b, a = butter(order, [lo/(fs/2), hi/(fs/2)], btype='band'); return b, a

def detect(x, fs=100, lo=8, hi=30, W_ms=70, beta=0.35, refr_ms=200, lam=0.125, searchback=True, return_feat=False):
    x = np.asarray(x, float); x = x - np.nanmean(x); x[np.isnan(x)] = 0
    b, a = design(fs, lo, hi)
    zi = lfilter_zi(b, a) * x[0]
    xb, _ = lfilter(b, a, x, zi=zi)                       # (1) bandpass, causal
    d = np.r_[0, np.diff(xb)]                             # (2) derivative
    s = d * d                                             # (3) square
    W = max(2, int(round(W_ms / 1000 * fs)))
    m = np.convolve(s, np.ones(W) / W, mode='full')[:len(s)]   # (4) moving-window integration (causal)
    refr = int(round(refr_ms / 1000 * fs))
    # (5) adaptive threshold (Pan-Tompkins style, single feature signal)
    SPK = np.max(m[:2 * fs]) * 0.5 if len(m) > 2 * fs else np.max(m) * 0.5
    NPK = np.mean(m[:2 * fs]) if len(m) > 2 * fs else np.mean(m)
    peaks = []; rr_buf = []; last = -refr; n = len(m)
    i = 1
    while i < n - 1:
        if m[i] > m[i - 1] and m[i] >= m[i + 1] and i - last > refr:     # local max candidate
            thr = NPK + beta * (SPK - NPK)
            if m[i] > thr:
                peaks.append(i); SPK = lam * m[i] + (1 - lam) * SPK
                if len(peaks) > 1: rr_buf.append(peaks[-1] - peaks[-2]); rr_buf = rr_buf[-8:]
                last = i
            else:
                NPK = lam * m[i] + (1 - lam) * NPK
                # search-back: if no beat for 1.66 x mean RR, accept the largest candidate above half threshold
                if searchback and rr_buf and i - last > 1.66 * np.mean(rr_buf):
                    seg0 = last + refr; seg = m[seg0:i + 1]
                    if len(seg) and seg.max() > 0.5 * thr:
                        j = seg0 + int(np.argmax(seg)); peaks.append(j); SPK = lam * m[j] + (1 - lam) * SPK
                        rr_buf.append(peaks[-1] - peaks[-2]); rr_buf = rr_buf[-8:]; last = j
        i += 1
    peaks = np.array(peaks, int)
    # (6) refine: R time = largest |xb| within +-60 ms before the integrator peak (integration delay ~ W/2 + filter delay)
    r = []
    for p in peaks:
        a0, a1 = max(0, p - int(0.12 * fs)), min(len(xb), p + int(0.02 * fs))
        r.append(a0 + int(np.argmax(np.abs(xb[a0:a1]))))
    r = np.array(sorted(set(r)), int)
    if return_feat: return r, xb, m
    return r

def rr_quality(r, fs=100):
    """Reference-free HR reliability from one detector (for the app): HR range + RR regularity."""
    if len(r) < 4: return dict(hr=np.nan, rrcv=np.nan, ok=False)
    rr = np.diff(r) / fs; hr = 60 / np.median(rr); cv = np.std(rr) / np.mean(rr)
    return dict(hr=hr, rrcv=cv, ok=bool(80 <= hr <= 220 and cv < 0.25))
