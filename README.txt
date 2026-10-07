NeoTex Beat Viewer
==================
Standalone app: load any ECG recording, detect every beat with the NeoTex belt detector v2, and play it back live
with beat markers, beat count, 10-s heart rate and the RR reliability gate.

Start
  python beat_viewer.py     (any Python 3.10+, e.g. your global 3.14, or the VS Code Run button)
  or double-click "Start Beat Viewer.cmd" (or drag a CSV onto it).
  If numpy/pandas/scipy/matplotlib are missing, the app installs requirements.txt into that same Python on first run.
  Manual install:  python -m pip install -r requirements.txt

Files
  beat_viewer.py              the app
  neotex_qrs_v2.py            detector v2 (copy from NeoTex Pipeline v2, 2026-10-06)
  neotex_qrs_params_v2.json   v2 paper parameters (lo 5, hi 25, W 50 ms, beta 0.25, clip 4, tau 1 s, dead 2 s)
  neotex_qrs.py + neotex_qrs_params.json   detector v1, used only for the "Overlay v1" comparison
  If the detector is changed in NeoTex Pipeline v2, copy those files here to keep the app in step.

Formats
  BELT CSV (time_s + ECG, 100 Hz) - detected directly at 100 Hz.
  BIOPAC CSV (AHA ECG column, 1000 Hz) - resampled to 100 Hz for detection; markers mapped back to 1000 Hz.
  Any other CSV/TXT/TSV/XLSX - ECG and time columns are guessed; change the column in the "Signal" box.
  No time column -> you are asked for the sampling rate.

Controls
  Space play/pause | Left/Right 1 s (Shift = 10 s) | Home = start | +/- speed | mouse wheel on ECG = window length
  Click the HR trend to jump. "Snap to R-peak" moves each detector time to the visible R-peak (+-50 ms);
  the export keeps both the snapped time and the raw detector time.

Export columns
  beat, time_s, sample_index, detector_sample, detector_time_s, rr_s, hr_bpm, window_hr_bpm, window_gate_ok
  Gate = HR 80-220 bpm and RR-CV < 0.25 in non-overlapping 10-s windows (same rule as rr_quality in neotex_qrs_v2).
