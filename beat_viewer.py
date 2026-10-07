#!/usr/bin/env python3
"""NeoTex Beat Viewer - load any ECG recording, detect every beat with the belt detector v2, and play it back live.

    python beat_viewer.py                 (then File > Open, or the Open button)
    python beat_viewer.py <file.csv>      (opens the file straight away)

Formats recognised automatically
    BELT    *_BELT_*.csv   columns time_s + ECG                    (100 Hz, the format in 'Organized Study Data')
    BIOPAC  *_BIOPAC_*.csv columns time_s/milliSec + AHA ECG column (1000 Hz; resampled to 100 Hz for detection)
    Other   any .csv/.txt/.tsv with a numeric column; the ECG column and the time column are guessed and can be changed
            in the 'Signal' box. Without a time column you are asked for the sampling rate.

Detection is the exact detector used in the paper pipeline: neotex_qrs_v2.detect() with neotex_qrs_params_v2.json. This folder
is standalone: it carries its own copy of neotex_qrs_v2.py / neotex_qrs.py (v1, for the overlay) and their parameter files, copied
from 'NeoTex Pipeline v2' on 2026-10-06. If the detector changes there, copy the new files here. The detector is causal, so the beats shown during live playback are the ones a streaming device would have found up to
that moment. Detector times are then snapped to the R-peak of the displayed ECG (+-50 ms, dominant polarity) so the marker sits
on the visible peak; both times are written when you export.

Keys: Space play/pause - Left/Right 1 s (Shift 10 s) - Home start - +/- speed - mouse wheel on the ECG changes the window.
"""
import os, sys, json, time, threading, queue, traceback

# --- missing packages: install them into the Python that started this script (global install), then continue ------------
def _bootstrap():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        import numpy, pandas, scipy, matplotlib  # noqa: F401
        return
    except ImportError as e:
        missing = e.name
    req = os.path.join(here, 'requirements.txt')
    print(f'Package "{missing}" is missing for {sys.executable}.\nInstalling requirements.txt into this Python (one time, needs internet)...', flush=True)
    import subprocess
    base = [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check', '-r', req]
    if subprocess.call(base) != 0 and subprocess.call(base + ['--user']) != 0 and subprocess.call(base + ['--break-system-packages']) != 0:
        sys.exit(f'Install failed. Run manually:\n  "{sys.executable}" -m pip install -r "{req}"\n'
                 f'or, for a uv-managed Python:\n  uv pip install --system --python "{sys.executable}" -r "{req}"')
    sys.exit(subprocess.call([sys.executable, os.path.abspath(__file__)] + sys.argv[1:]))
_bootstrap()
from fractions import Fraction
import numpy as np, pandas as pd
from scipy import signal
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import neotex_qrs_v2 as v2
try: import neotex_qrs as v1
except ImportError: v1 = None

DET_FS = 100                       # detector is tuned and validated at 100 Hz
V2_DEFAULTS = dict(lo=5, hi=25, W_ms=50, beta=0.25, clip=4.0, tau_s=1.0, dead_s=2.0, refr_ms=200)
V1_DEFAULTS = dict(lo=5, hi=25, W_ms=50, beta=0.25)
GATE_WIN, HR_MIN, HR_MAX, RRCV_MAX = 10.0, 80, 220, 0.25     # same constants as config.py / rr_quality()
SNAP_S = 0.05

def load_json(name, default):
    p = os.path.join(HERE, name)
    try:
        with open(p) as f: d = json.load(f)
        out = dict(default); out.update(d); return out
    except Exception: return dict(default)

# ----------------------------------------------------------------------------------------------------------------- loading
def _numeric_first_row(path, sep):
    with open(path, 'r', errors='ignore') as f: first = f.readline().strip()
    toks = [x for x in (first.split() if sep == r'\s+' else first.split(sep)) if x.strip()]
    try: [float(x) for x in toks]; return len(toks)
    except ValueError: return 0

def read_table(path, nrows=None, usecols=None):
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.xlsx', '.xls'): return pd.read_excel(path, nrows=nrows, usecols=usecols)
    if ext == '.csv': sep = ','
    elif ext == '.tsv': sep = '\t'
    else:
        with open(path, 'r', errors='ignore') as f: head = [f.readline() for _ in range(5)]   # pick the separator that is used
        counts = {s_: min(l.count(s_) for l in head if l.strip()) for s_ in (',', '\t', ';', '|')}
        sep = max(counts, key=counts.get)
        if counts[sep] == 0: sep = r'\s+' if any(len(l.split()) > 1 for l in head if l.strip()) else ','
    kw = dict(sep=sep, nrows=nrows, engine='python' if sep == r'\s+' else 'c')
    ncol = _numeric_first_row(path, sep)
    if ncol:                                   # no header row: name the columns col1, col2, ...
        kw.update(header=None, names=[f'col{i + 1}' for i in range(ncol)])
    return pd.read_csv(path, usecols=usecols, **kw)

def inspect(path):
    """Return (kind, ecg_candidates, default_ecg_col, time_col). Candidates are numeric columns ordered by plausibility."""
    h = read_table(path, nrows=200)
    if h.shape[1] == 1 and not pd.api.types.is_numeric_dtype(h.iloc[:, 0]):
        raise ValueError('Could not find numeric columns in this file.')
    cols = list(h.columns)
    num = [c for c in cols if pd.api.types.is_numeric_dtype(h[c])]
    low = {c: str(c).lower() for c in cols}
    tcol = next((c for c in ('time_orig_s', 'time_s', 'time', 'Time', 't') if c in cols), None)
    if tcol is None: tcol = next((c for c in num if 'time' in low[c] or low[c] in ('sec', 'seconds')), None)
    if tcol is None and 'milliSec' in cols: tcol = 'milliSec'
    aha = [c for c in cols if 'AHA' in str(c)]; mod = [c for c in cols if 'ECG MODULE' in str(c)]
    if aha or mod:
        kind = 'BIOPAC'; default = (aha or mod)[0]
        ecg_c = aha + [c for c in mod if c not in aha]
    elif 'ECG' in cols:
        kind = 'BELT' if ('Tx_ms' in cols or 'AccX' in cols or 'BELT' in os.path.basename(path).upper()) else 'ECG file'
        default = 'ECG'; ecg_c = ['ECG']
    else:
        kind = 'ECG file'; ecg_c = [c for c in num if 'ecg' in low[c] or 'lead' in low[c] or low[c] in ('mlii', 'ii', 'v1', 'v5')]
        default = ecg_c[0] if ecg_c else None
    skip = {tcol, 'sample_index', 'milliSec', 'Tx_ms', 'Rx_ms', 'Latency', 'InterArrival', 'Seq', 'PC_Time'}
    rest = [c for c in num if c not in ecg_c and c not in skip]
    cands = ecg_c + rest
    if default is None: default = cands[0] if cands else None
    if default is None: raise ValueError('No numeric signal column found.')
    return kind, cands, default, tcol

def load_signal(path, col, tcol, fs_hint=None):
    use = [col] + ([tcol] if tcol and tcol != col else [])
    d = read_table(path, usecols=use)
    x = pd.to_numeric(d[col], errors='coerce').astype(float)
    if x.isna().all(): raise ValueError(f'Column "{col}" has no numeric values.')
    x = x.interpolate(limit_direction='both').values
    if tcol:
        t = pd.to_numeric(d[tcol], errors='coerce').astype(float).interpolate(limit_direction='both').values
        if tcol == 'milliSec': t = t / 1000.0
        dt = np.diff(t); dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0: raise ValueError(f'Time column "{tcol}" is not increasing.')
        fs = 1.0 / np.median(dt)
        if abs(fs - round(fs)) < 0.02 * fs: fs = float(round(fs))
        jitter = float(np.std(dt) / np.mean(dt))
        t = t[0] + np.arange(len(x)) / fs          # uniform axis (the detector assumes uniform sampling)
    else:
        fs = float(fs_hint or 100); t = np.arange(len(x)) / fs; jitter = 0.0
    return t, x, fs, jitter

# --------------------------------------------------------------------------------------------------------------- detection
def bandpass(x, lo, hi, fs, order=2):
    hi = min(hi, 0.45 * fs)
    x = np.asarray(x, float) - np.nanmedian(x)
    return signal.sosfiltfilt(signal.butter(order, [lo, hi], 'bp', fs=fs, output='sos'), x)

def to_det_rate(x, fs):
    if abs(fs - DET_FS) < 1e-6: return np.asarray(x, float)
    fr = Fraction(DET_FS / fs).limit_denominator(1000)
    return signal.resample_poly(np.asarray(x, float) - np.median(x), fr.numerator, fr.denominator)

def run_detector(mod, x, fs, params):
    """Run a belt detector at 100 Hz; return beat sample indices at the native rate (before snapping)."""
    xd = to_det_rate(x, fs)
    keys = mod.detect.__code__.co_varnames[:mod.detect.__code__.co_argcount]
    P = {k: v for k, v in params.items() if k in keys}
    r = np.asarray(mod.detect(xd, DET_FS, **P), int)
    return np.clip(np.round(r * fs / DET_FS).astype(int), 0, len(x) - 1)

def snap(idx, disp, fs, win_s=SNAP_S):
    """Move each detector time to the dominant-polarity extremum of the displayed ECG within +-win_s."""
    if len(idx) == 0: return idx, 1
    w = max(1, int(round(win_s * fs)))
    amp = np.array([disp[max(0, i - w):i + w + 1][np.argmax(np.abs(disp[max(0, i - w):i + w + 1]))] for i in idx])
    pol = 1 if np.median(amp) >= 0 else -1
    out = np.array([max(0, i - w) + int(np.argmax(pol * disp[max(0, i - w):i + w + 1])) for i in idx], int)
    out = np.unique(out)
    return out, pol

def rr_gate(bt):
    """bt: beat times (s) inside one window -> (hr, rrcv, ok). Same rule as neotex_qrs_v2.rr_quality."""
    if len(bt) < 4: return np.nan, np.nan, False
    rr = np.diff(bt); hr = 60 / np.median(rr); cv = np.std(rr) / np.mean(rr)
    return hr, cv, bool(HR_MIN <= hr <= HR_MAX and cv < RRCV_MAX)

# ---------------------------------------------------------------------------------------------------------------------- GUI
C = dict(bg='#f6f7f9', panel='#ffffff', ink='#1d2430', mute='#6b7480', ecg='#1f3b5c', beat='#d1342f', v1='#8a94a3',
         ok='#1e8a4c', bad='#c0392b', warn='#b7791f', grid='#e3e6ea', failbg='#fdf1f0', span='#2f6fdd')

class Viewer(tk.Tk):
    def __init__(self, path=None):
        super().__init__()
        self.title('NeoTex Beat Viewer - detector v2'); sw, sh = self.winfo_screenwidth(), self.winfo_screenheight(); self.geometry(f'{min(sw - 40, 1440)}x{min(sh - 90, 900)}+10+10'); self.minsize(1100, 680); self.configure(bg=C['bg'])
        self.report_callback_exception = self._tk_error
        self.params = load_json('neotex_qrs_params_v2.json', V2_DEFAULTS); self.file_params = dict(self.params)
        self.v1_params = load_json('neotex_qrs_params.json', V1_DEFAULTS)
        self.rec = None; self.playing = False; self.cur = 0.0; self.last_tick = None; self.q = queue.Queue(); self.busy = False
        self.ylim = None; self._slider_lock = False
        self._style(); self._build(); self._bind_keys()
        self.after(60, self._pump); self.after(40, self._tick)
        if path: self.after(200, lambda: self.open_file(path))

    # ---------- layout
    def _style(self):
        s = ttk.Style(self)
        try: s.theme_use('clam')
        except tk.TclError: pass
        base = ('Segoe UI', 10) if sys.platform == 'win32' else ('Helvetica', 10)
        self.f_base = base
        s.configure('.', font=base, background=C['bg'], foreground=C['ink'])
        s.configure('Panel.TFrame', background=C['panel']); s.configure('Panel.TLabel', background=C['panel'])
        s.configure('Mute.TLabel', background=C['panel'], foreground=C['mute'], font=(base[0], 9))
        s.configure('Big.TLabel', background=C['panel'], font=(base[0], 26, 'bold'))
        s.configure('Mid.TLabel', background=C['panel'], font=(base[0], 13, 'bold'))
        s.configure('Head.TLabel', background=C['panel'], foreground=C['mute'], font=(base[0], 9, 'bold'))
        s.configure('Accent.TButton', font=(base[0], 10, 'bold'))
        s.configure('TLabelframe', background=C['panel']); s.configure('TLabelframe.Label', background=C['panel'], foreground=C['mute'])
        s.configure('TCheckbutton', background=C['bg']); s.configure('Panel.TCheckbutton', background=C['panel'])

    def _build(self):
        top = ttk.Frame(self, padding=(10, 8, 10, 4)); top.pack(fill='x')
        ttk.Button(top, text='Open ECG file…', style='Accent.TButton', command=self.open_dialog).pack(side='left')
        self.v_file = tk.StringVar(value='No file loaded'); ttk.Label(top, textvariable=self.v_file, width=44).pack(side='left', padx=10)
        ttk.Label(top, text='Signal').pack(side='left')
        self.v_col = tk.StringVar(); self.cb_col = ttk.Combobox(top, textvariable=self.v_col, width=30, state='readonly')
        self.cb_col.pack(side='left', padx=(4, 10)); self.cb_col.bind('<<ComboboxSelected>>', lambda e: self.reload_column())
        self.v_v1 = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text='Overlay v1', variable=self.v_v1, command=self.toggle_v1, state='normal' if v1 else 'disabled').pack(side='left')
        self.v_snap = tk.BooleanVar(value=True)
        ttk.Checkbutton(top, text='Snap to R-peak', variable=self.v_snap, command=self.redetect).pack(side='left', padx=8)
        ttk.Button(top, text='Export beats…', command=self.export).pack(side='right')
        ttk.Button(top, text='Save view PNG', command=self.save_png).pack(side='right', padx=6)

        ctl = ttk.Frame(self, padding=(10, 0, 10, 6)); ctl.pack(fill='x')
        self.b_play = ttk.Button(ctl, text='▶  Play', width=10, command=self.toggle_play, style='Accent.TButton'); self.b_play.pack(side='left')
        ttk.Button(ctl, text='⏮', width=3, command=lambda: self.seek(0)).pack(side='left', padx=(4, 0))
        ttk.Button(ctl, text='−10 s', command=lambda: self.seek(self.cur - 10)).pack(side='left', padx=(4, 0))
        ttk.Button(ctl, text='+10 s', command=lambda: self.seek(self.cur + 10)).pack(side='left', padx=(2, 10))
        ttk.Label(ctl, text='Speed').pack(side='left')
        self.v_speed = tk.StringVar(value='1×')
        ttk.Combobox(ctl, textvariable=self.v_speed, width=5, state='readonly', values=['0.25×', '0.5×', '1×', '2×', '4×', '8×', '16×', '32×']).pack(side='left', padx=(4, 10))
        ttk.Label(ctl, text='Window').pack(side='left')
        self.v_win = tk.StringVar(value='8 s')
        cw = ttk.Combobox(ctl, textvariable=self.v_win, width=5, state='readonly', values=['3 s', '5 s', '8 s', '10 s', '15 s', '30 s', '60 s'])
        cw.pack(side='left', padx=(4, 10)); cw.bind('<<ComboboxSelected>>', lambda e: self.render())
        self.v_live = tk.BooleanVar(value=True)
        ttk.Checkbutton(ctl, text='Live sweep (hide samples after the cursor)', variable=self.v_live, command=self.render).pack(side='left')
        self.v_time = tk.StringVar(value='0:00.0 / 0:00.0'); ttk.Label(ctl, textvariable=self.v_time, font=(self.f_base[0], 10, 'bold')).pack(side='right')
        self.slider = ttk.Scale(self, from_=0, to=1, orient='horizontal', command=self._on_slider); self.slider.pack(fill='x', padx=10)

        self.v_status = tk.StringVar(value='Open a BELT, BIOPAC or any ECG CSV to start.  Space = play/pause.')
        ttk.Label(self, textvariable=self.v_status, padding=(10, 2, 10, 6), foreground=C['mute']).pack(side='bottom', fill='x')
        body = ttk.Frame(self, padding=(10, 6, 10, 6)); body.pack(fill='both', expand=True)
        body.columnconfigure(0, weight=1); body.rowconfigure(0, weight=1)
        figf = ttk.Frame(body, style='Panel.TFrame'); figf.grid(row=0, column=0, sticky='nsew')
        self.fig = Figure(figsize=(10, 6), dpi=100, facecolor=C['panel'])
        gs = self.fig.add_gridspec(2, 1, height_ratios=[3.2, 1], hspace=0.32, left=0.07, right=0.985, top=0.95, bottom=0.08)
        self.ax = self.fig.add_subplot(gs[0]); self.ax2 = self.fig.add_subplot(gs[1])
        self.canvas = FigureCanvasTkAgg(self.fig, master=figf); self.canvas.get_tk_widget().pack(fill='both', expand=True)
        self._init_axes()

        side = ttk.Frame(body, style='Panel.TFrame', padding=14, width=270); side.grid(row=0, column=1, sticky='ns', padx=(10, 0))
        side.grid_propagate(False)
        ttk.Label(side, text='BEATS DETECTED', style='Head.TLabel').pack(anchor='w')
        self.v_count = tk.StringVar(value='0'); ttk.Label(side, textvariable=self.v_count, style='Big.TLabel', foreground=C['beat']).pack(anchor='w')
        self.v_count_tot = tk.StringVar(value='of 0 in the recording'); ttk.Label(side, textvariable=self.v_count_tot, style='Mute.TLabel').pack(anchor='w', pady=(0, 10))
        ttk.Label(side, text='HEART RATE · LAST 10 s', style='Head.TLabel').pack(anchor='w')
        self.v_hr = tk.StringVar(value='—'); self.l_hr = ttk.Label(side, textvariable=self.v_hr, style='Big.TLabel'); self.l_hr.pack(anchor='w')
        self.v_gate = tk.StringVar(value='RR gate: —'); self.l_gate = ttk.Label(side, textvariable=self.v_gate, style='Panel.TLabel', font=(self.f_base[0], 10, 'bold'))
        self.l_gate.pack(anchor='w')
        self.v_gate2 = tk.StringVar(value=''); ttk.Label(side, textvariable=self.v_gate2, style='Mute.TLabel').pack(anchor='w', pady=(0, 10))
        ttk.Label(side, text='LAST BEAT', style='Head.TLabel').pack(anchor='w')
        self.v_rr = tk.StringVar(value='—'); ttk.Label(side, textvariable=self.v_rr, style='Mid.TLabel').pack(anchor='w', pady=(0, 10))
        ttk.Separator(side).pack(fill='x', pady=6)
        self.recf = ttk.Frame(side, style='Panel.TFrame'); self.recf.pack(fill='x')
        ttk.Label(self.recf, text='RECORDING', style='Head.TLabel').pack(anchor='w')
        self.v_info = tk.StringVar(value='—'); ttk.Label(self.recf, textvariable=self.v_info, style='Mute.TLabel', justify='left', wraplength=245).pack(anchor='w', pady=(2, 8))
        self.sep2 = ttk.Separator(side); self.sep2.pack(fill='x', pady=6)
        self.b_adv = ttk.Button(side, text='Detector v2 parameters  ▸', command=self.toggle_adv); self.b_adv.pack(fill='x')
        adv = ttk.LabelFrame(side, text='from neotex_qrs_params_v2.json', padding=8); self.adv = adv
        self.pvars = {}
        for i, (k, lab) in enumerate([('lo', 'Low Hz'), ('hi', 'High Hz'), ('W_ms', 'W ms'), ('beta', 'β'),
                                       ('refr_ms', 'Refr ms'), ('clip', 'Clip ×'), ('tau_s', 'τ s'), ('dead_s', 'Dead s')]):
            r_, c_ = i // 2, (i % 2) * 2
            ttk.Label(adv, text=lab, style='Panel.TLabel').grid(row=r_, column=c_, sticky='w', padx=(0 if c_ == 0 else 8, 2))
            v = tk.StringVar(value=str(self.params.get(k, V2_DEFAULTS[k]))); self.pvars[k] = v
            ttk.Entry(adv, textvariable=v, width=6).grid(row=r_, column=c_ + 1, sticky='e', pady=1)
        bf = ttk.Frame(adv, style='Panel.TFrame'); bf.grid(row=99, column=0, columnspan=4, sticky='ew', pady=(6, 0))
        ttk.Button(bf, text='Apply', command=self.apply_params).pack(side='left')
        ttk.Button(bf, text='Reset to paper values', command=self.reset_params).pack(side='right')


    def _init_axes(self):
        ax, ax2 = self.ax, self.ax2
        for a in (ax, ax2):
            a.set_facecolor(C['panel']); a.grid(True, color=C['grid'], lw=0.7); a.tick_params(colors=C['mute'], labelsize=8.5)
            for s in a.spines.values(): s.set_color(C['grid'])
        ax.set_ylabel('ECG', color=C['mute'], fontsize=9); ax2.set_ylabel('HR (bpm)', color=C['mute'], fontsize=9)
        ax2.set_xlabel('time (s)', color=C['mute'], fontsize=9)
        self.l_ecg, = ax.plot([], [], color=C['ecg'], lw=0.9)
        self.l_beat, = ax.plot([], [], ls='none', marker='o', ms=6.5, mfc=C['beat'], mec='white', mew=0.8, zorder=5)
        self.l_v1, = ax.plot([], [], ls='none', marker='x', ms=6, color=C['v1'], mew=1.4, zorder=4)
        self.l_cur = ax.axvline(0, color=C['span'], lw=1.0, alpha=0.6)
        self.texts = [ax.text(0, 0, '', fontsize=7.5, color=C['beat'], ha='center', va='bottom', zorder=6, clip_on=True) for _ in range(80)]
        self.t_title = ax.set_title('', loc='left', fontsize=10, color=C['ink'])
        self.l_hrdots, = ax2.plot([], [], ls='none', marker='.', ms=2.5, color=C['mute'], alpha=0.55)
        self.l_hrok, = ax2.plot([], [], color=C['ok'], lw=2.0, solid_capstyle='butt')
        self.l_hrbad, = ax2.plot([], [], color=C['bad'], lw=1.2, alpha=0.6, solid_capstyle='butt')
        self.l_cur2 = ax2.axvline(0, color=C['span'], lw=1.2)
        self.r_span = Rectangle((0, 0), 0, 1, transform=ax2.get_xaxis_transform(), color=C['span'], alpha=0.12, lw=0); ax2.add_patch(self.r_span)
        ax.text(0.5, 0.5, 'Open an ECG file', transform=ax.transAxes, ha='center', va='center', color=C['mute'], fontsize=13)
        self.canvas.draw_idle()

    def _bind_keys(self):
        self.bind('<space>', lambda e: self.toggle_play()); self.bind('<Home>', lambda e: self.seek(0))
        self.bind('<Left>', lambda e: self.seek(self.cur - 1)); self.bind('<Right>', lambda e: self.seek(self.cur + 1))
        self.bind('<Shift-Left>', lambda e: self.seek(self.cur - 10)); self.bind('<Shift-Right>', lambda e: self.seek(self.cur + 10))
        self.bind('<plus>', lambda e: self.step_speed(1)); self.bind('<equal>', lambda e: self.step_speed(1)); self.bind('<minus>', lambda e: self.step_speed(-1))
        self.canvas.mpl_connect('button_press_event', self._on_click); self.canvas.mpl_connect('scroll_event', self._on_scroll)
        self.protocol('WM_DELETE_WINDOW', self.destroy)

    # ---------- loading / detection (worker thread)
    def open_dialog(self):
        study = os.path.join(os.path.dirname(HERE), 'Organized Study Data')
        p = filedialog.askopenfilename(title='Open ECG recording', filetypes=[('ECG data', '*.csv *.txt *.tsv *.xlsx'), ('All files', '*.*')],
                                       initialdir=os.path.dirname(self.rec['path']) if self.rec else (study if os.path.isdir(study) else HERE))
        if p: self.open_file(p)

    def open_file(self, path):
        if self.busy: return
        try: kind, cands, default, tcol = inspect(path)
        except Exception as e: messagebox.showerror('Cannot read file', f'{os.path.basename(path)}\n\n{e}'); return
        fs_hint = None
        if tcol is None:
            fs_hint = simpledialog.askfloat('Sampling rate', 'No time column found.\nSampling rate of this file (Hz):', initialvalue=100, minvalue=20, maxvalue=20000, parent=self)
            if not fs_hint: return
        self.meta = dict(path=path, kind=kind, cands=cands, tcol=tcol, fs_hint=fs_hint)
        self.cb_col.configure(values=[str(c) for c in cands]); self.v_col.set(str(default))
        self._start_load(default)

    def reload_column(self):
        if not getattr(self, 'meta', None) or self.busy: return
        col = next(c for c in self.meta['cands'] if str(c) == self.v_col.get()); self._start_load(col)

    def _start_load(self, col):
        self.pause(); self.busy = True; m = self.meta
        self.v_status.set(f'Loading {os.path.basename(m["path"])} …'); self.config(cursor='watch')
        params = self.params.copy(); want_v1 = self.v_v1.get(); snap_on = self.v_snap.get()
        def work():
            try:
                t, x, fs, jit = load_signal(m['path'], col, m['tcol'], m['fs_hint'])
                self.q.put(('status', f'Detecting beats (v2) on {len(x):,} samples …'))
                rec = dict(path=m['path'], kind=m['kind'], col=col, t=t, raw=x, fs=fs, jitter=jit)
                self._detect_into(rec, params, want_v1, snap_on)
                self.q.put(('loaded', rec))
            except Exception as e:
                self.q.put(('error', f'{e}\n\n{traceback.format_exc(limit=3)}'))
        threading.Thread(target=work, daemon=True).start()

    def _detect_into(self, rec, params, want_v1, snap_on):
        fs, x = rec['fs'], rec['raw']
        lo, hi = (2, 35) if fs <= 250 else (0.5, 40)        # display bands = neotex_signals.ecg_belt / ecg_ref
        rec['disp'] = bandpass(x, lo, hi, fs); rec['band'] = (lo, hi)
        t0 = time.perf_counter()
        det = run_detector(v2, x, fs, params)
        rec['det_idx'] = det
        if snap_on: idx, pol = snap(det, rec['disp'], fs)
        else: idx, pol = det, 1
        rec['idx'] = idx; rec['pol'] = pol; rec['bt'] = rec['t'][idx]; rec['det_ms'] = (time.perf_counter() - t0) * 1000
        rec['v1_bt'] = None
        if want_v1 and v1 is not None:
            d1 = run_detector(v1, x, fs, self.v1_params)
            rec['v1_bt'] = rec['t'][snap(d1, rec['disp'], fs)[0] if snap_on else d1]
        self._summaries(rec)

    def _summaries(self, rec):
        bt = rec['bt']; t = rec['t']
        rr = np.diff(bt); rec['ihr'] = np.r_[np.nan, 60 / rr] if len(bt) else np.array([])
        edges = np.arange(t[0], t[-1] + 1e-9, GATE_WIN)
        wins = []
        for a in edges:
            b = a + GATE_WIN
            if b > t[-1] + 1e-9: break
            hr, cv, ok = rr_gate(bt[(bt >= a) & (bt < b)]); wins.append((a, hr, cv, ok))
        rec['wins'] = wins
        ok = [w for w in wins if w[3]]
        rec['pct_ok'] = 100 * len(ok) / len(wins) if wins else np.nan
        rec['hr_ok'] = np.median([w[1] for w in ok]) if ok else np.nan

    def _pump(self):
        try:
            while True:
                kind, val = self.q.get_nowait()
                if kind == 'status': self.v_status.set(val)
                elif kind == 'error':
                    self.busy = False; self.config(cursor=''); self.v_status.set('Error - see dialog.'); messagebox.showerror('Error', val)
                elif kind in ('loaded', 'redetected'):
                    self.busy = False; self.config(cursor=''); keep = kind == 'redetected'
                    self.rec = val
                    if not keep: self.cur = val['t'][0] + min(self._win(), val['t'][-1] - val['t'][0]); self.ylim = None
                    self._after_load(keep)
        except queue.Empty: pass
        self.after(60, self._pump)

    def _after_load(self, keep):
        r = self.rec; dur = r['t'][-1] - r['t'][0]
        self.v_file.set(f'{os.path.basename(r["path"])}   [{r["kind"]}]')
        self.slider.configure(from_=r['t'][0], to=r['t'][-1])
        resamp = '' if abs(r['fs'] - DET_FS) < 1e-6 else f'\nDetector input resampled {r["fs"]:g}→{DET_FS} Hz'
        jit = f'\nTime-stamp jitter {100 * r["jitter"]:.0f}% (uniform axis used)' if r['jitter'] > 0.05 else ''
        note = '' if r['kind'] == 'BELT' else '\nv2 was tuned on belt ECG; check markers on other sources.'
        self.v_info.set(f'{fmt(dur)} · {r["fs"]:g} Hz · {len(r["raw"]):,} samples\nColumn: {r["col"]}\nDisplay band {r["band"][0]}–{r["band"][1]} Hz'
                        f'{resamp}\nReliable 10-s windows: {r["pct_ok"]:.0f}%'
                        + (f'  (median HR {r["hr_ok"]:.0f} bpm)' if np.isfinite(r['hr_ok']) else '') + jit + note)
        self.v_count_tot.set(f'of {len(r["bt"]):,} in the recording')
        self._draw_trend()
        self.v_status.set(f'{len(r["bt"]):,} beats detected with v2 in {r["det_ms"]:.0f} ms.  Space = play/pause · ←/→ step · wheel = window length.')
        self.render()

    def _params_from_ui(self):
        P = dict(self.params)
        for k, v in self.pvars.items():
            try: P[k] = float(v.get()) if k not in ('W_ms', 'refr_ms') else int(float(v.get()))
            except ValueError: raise ValueError(f'Parameter {k} must be a number.')
        if not 0 < P['lo'] < P['hi'] < DET_FS / 2: raise ValueError(f'Band must satisfy 0 < low < high < {DET_FS / 2:g} Hz (detector runs at {DET_FS} Hz).')
        return P

    def toggle_adv(self):
        if self.adv.winfo_ismapped():
            self.adv.pack_forget(); self.recf.pack(fill='x', before=self.sep2); self.b_adv.configure(text='Detector v2 parameters  ▸')
        else:
            self.recf.pack_forget(); self.adv.pack(fill='x', pady=(6, 0)); self.b_adv.configure(text='Detector v2 parameters  ▾')

    def apply_params(self):
        try: self.params = self._params_from_ui()
        except ValueError as e: messagebox.showerror('Parameters', str(e)); return
        self.redetect()

    def reset_params(self):
        self.params = dict(self.file_params)
        for k, v in self.pvars.items(): v.set(str(self.params.get(k, V2_DEFAULTS[k])))
        self.redetect()

    def toggle_v1(self): self.redetect()

    def redetect(self):
        if not self.rec or self.busy: return
        self.busy = True; self.config(cursor='watch'); self.v_status.set('Re-running detector …')
        rec = self.rec.copy(); params = self.params.copy(); want_v1 = self.v_v1.get(); snap_on = self.v_snap.get()
        def work():
            try: self._detect_into(rec, params, want_v1, snap_on); self.q.put(('redetected', rec))
            except Exception as e: self.q.put(('error', f'{e}\n\n{traceback.format_exc(limit=3)}'))
        threading.Thread(target=work, daemon=True).start()

    # ---------- playback
    def _speed(self):
        try: return float(self.v_speed.get().rstrip('×'))
        except ValueError: return 1.0
    def _win(self):
        try: return float(self.v_win.get().split()[0])
        except ValueError: return 8.0
    def step_speed(self, d):
        vals = ['0.25×', '0.5×', '1×', '2×', '4×', '8×', '16×', '32×']
        i = vals.index(self.v_speed.get()) if self.v_speed.get() in vals else 2
        self.v_speed.set(vals[max(0, min(len(vals) - 1, i + d))])
    def toggle_play(self):
        if not self.rec: return
        if self.playing: self.pause()
        else:
            if self.cur >= self.rec['t'][-1] - 1e-6: self.cur = self.rec['t'][0]
            self.playing = True; self.last_tick = time.perf_counter(); self.b_play.configure(text='❚❚  Pause')
    def pause(self):
        self.playing = False; self.b_play.configure(text='▶  Play')
    def seek(self, t):
        if not self.rec: return
        self.cur = float(np.clip(t, self.rec['t'][0], self.rec['t'][-1])); self.render()

    def _tick(self):
        frame_ms = 40
        if self.playing and self.rec:
            now = time.perf_counter(); dt = now - (self.last_tick or now); self.last_tick = now
            self.cur += dt * self._speed()
            if self.cur >= self.rec['t'][-1]: self.cur = self.rec['t'][-1]; self.pause()
            t0 = time.perf_counter(); self.render(); spent = (time.perf_counter() - t0) * 1000
            frame_ms = int(max(15, 40 - spent))
        self.after(frame_ms, self._tick)

    def _on_slider(self, v):
        if self._slider_lock or not self.rec: return
        self.cur = float(v); self.render(from_slider=True)
    def _on_click(self, ev):
        if self.rec and ev.inaxes is self.ax2 and ev.xdata is not None:
            self.seek(ev.xdata + (self._win() / 2 if not self.v_live.get() else 0))
    def _on_scroll(self, ev):
        if ev.inaxes is not self.ax: return
        vals = ['3 s', '5 s', '8 s', '10 s', '15 s', '30 s', '60 s']
        i = vals.index(self.v_win.get()) if self.v_win.get() in vals else 2
        self.v_win.set(vals[max(0, min(len(vals) - 1, i + (1 if ev.button == 'down' else -1)))]); self.render()

    # ---------- drawing
    def _draw_trend(self):
        r = self.rec; ax2 = self.ax2
        bt, ihr = r['bt'], r['ihr']
        m = np.isfinite(ihr) & (ihr > 30) & (ihr < 320)
        self.l_hrdots.set_data(bt[m], ihr[m])
        xs_ok, ys_ok, xs_bad, ys_bad = [], [], [], []
        for a, hr, cv, ok in r['wins']:
            if not np.isfinite(hr): continue
            xs, ys = (xs_ok, ys_ok) if ok else (xs_bad, ys_bad)
            xs.extend([a, a + GATE_WIN, np.nan]); ys.extend([hr, hr, np.nan])
        self.l_hrok.set_data(xs_ok, ys_ok); self.l_hrbad.set_data(xs_bad, ys_bad)
        ax2.set_xlim(r['t'][0], r['t'][-1])
        vals = ihr[m]
        lo, hi = (np.percentile(vals, 1), np.percentile(vals, 99)) if len(vals) > 5 else (60, 200)
        ax2.set_ylim(max(30, lo - 15), min(320, hi + 15))
        ax2.legend([self.l_hrok, self.l_hrbad, self.l_hrdots], ['10-s HR, gate pass', '10-s HR, gate fail', 'beat-to-beat'],
                   loc='upper right', fontsize=7.5, frameon=False, ncol=3)
        for txt in list(self.ax.texts):
            if txt not in self.texts: txt.remove()

    def render(self, from_slider=False):
        if not self.rec: return
        r = self.rec; t, y, fs = r['t'], r['disp'], r['fs']; W = self._win(); live = self.v_live.get(); cur = self.cur
        a, b = (cur - W, cur) if live else (cur - W / 2, cur + W / 2)
        end = min(b, cur) if live else b
        i0 = max(0, int((a - t[0]) * fs)); i1 = min(len(t), int((end - t[0]) * fs) + 1)
        tt, yy = t[i0:i1], y[i0:i1]
        step = max(1, len(tt) // 3000)
        self.l_ecg.set_data(tt[::step], yy[::step])
        ax = self.ax; ax.set_xlim(a, b)
        # y scale: robust, smoothed so it doesn't jump every frame
        seg = y[max(0, int((a - t[0]) * fs)):min(len(t), int((b - t[0]) * fs) + 1)]
        if len(seg) > 10:
            lo, hi = np.percentile(seg, [0.5, 99.5]); pad = 0.18 * (hi - lo + 1e-9); tgt = np.array([lo - pad, hi + pad * 1.8])
            self.ylim = tgt if self.ylim is None else 0.75 * self.ylim + 0.25 * tgt
            ax.set_ylim(*self.ylim)
        bt = r['bt']; shown_end = cur if live else b
        k0, k1 = np.searchsorted(bt, a), np.searchsorted(bt, shown_end, side='right')
        bidx = r['idx'][k0:k1]
        self.l_beat.set_data(t[bidx], y[bidx])
        yl = ax.get_ylim(); off = 0.04 * (yl[1] - yl[0])
        for j, txt in enumerate(self.texts):
            k = k0 + j
            if k < k1: txt.set_position((bt[k], y[r['idx'][k]] + off)); txt.set_text(str(k + 1)); txt.set_visible(True)
            else: txt.set_visible(False)
        if r.get('v1_bt') is not None:
            v = r['v1_bt']; v = v[(v >= a) & (v <= shown_end)]
            self.l_v1.set_data(v, np.full(len(v), yl[1] - 0.06 * (yl[1] - yl[0])))
        else: self.l_v1.set_data([], [])
        self.l_cur.set_xdata([cur, cur])
        # readouts
        n_done = int(np.searchsorted(bt, cur, side='right'))
        self.v_count.set(f'{n_done:,}')
        g = bt[(bt > cur - GATE_WIN) & (bt <= cur)]
        hr, cv, ok = rr_gate(g)
        if np.isfinite(hr):
            self.v_hr.set(f'{hr:.0f} bpm'); self.l_hr.configure(foreground=C['ok'] if ok else C['bad'])
            self.v_gate.set('RR gate: PASS - HR reliable' if ok else 'RR gate: FAIL - HR not reliable')
            self.l_gate.configure(foreground=C['ok'] if ok else C['bad'])
            self.v_gate2.set(f'{len(g)} beats · RR-CV {cv:.2f} (limit {RRCV_MAX}) · HR {HR_MIN}–{HR_MAX}')
        else:
            self.v_hr.set('—'); self.l_hr.configure(foreground=C['mute']); self.v_gate.set('RR gate: too few beats'); self.l_gate.configure(foreground=C['warn'])
            self.v_gate2.set(f'{len(g)} beats in the last {GATE_WIN:.0f} s')
        if n_done >= 2:
            rr = bt[n_done - 1] - bt[n_done - 2]; self.v_rr.set(f'#{n_done}  at {bt[n_done - 1]:.2f} s\nRR {rr * 1000:.0f} ms · {60 / rr:.0f} bpm')
        elif n_done == 1: self.v_rr.set(f'#1  at {bt[0]:.2f} s')
        else: self.v_rr.set('—')
        ax.set_facecolor(C['panel'] if (ok or not np.isfinite(hr)) else C['failbg'])
        self.t_title.set_text(f'{os.path.basename(r["path"])}  ·  {r["col"]}  ·  detector v2' + ('  ·  × = v1' if r.get('v1_bt') is not None else ''))
        self.l_cur2.set_xdata([cur, cur]); self.r_span.set_x(a); self.r_span.set_width(b - a)
        self.v_time.set(f'{fmt(cur - t[0])} / {fmt(t[-1] - t[0])}')
        if not from_slider:
            self._slider_lock = True; self.slider.set(cur); self._slider_lock = False
        self.canvas.draw_idle()

    # ---------- export
    def export(self):
        if not self.rec: return
        r = self.rec; stem = os.path.splitext(os.path.basename(r['path']))[0]
        p = filedialog.asksaveasfilename(title='Export beats', defaultextension='.csv', initialdir=os.path.dirname(r['path']),
                                         initialfile=f'{stem}_beats_v2.csv', filetypes=[('CSV', '*.csv')])
        if not p: return
        bt = r['bt']; rr = np.r_[np.nan, np.diff(bt)]
        det = r['det_idx']
        # map every exported beat to its raw detector sample (nearest)
        if len(det):
            j = np.clip(np.searchsorted(det, r['idx']), 1, len(det) - 1); j = np.where(np.abs(det[j - 1] - r['idx']) <= np.abs(det[j] - r['idx']), j - 1, j)
            dsamp = det[j]
        else: dsamp = np.array([], int)
        gate = np.zeros(len(bt), bool); whr = np.full(len(bt), np.nan)
        for a, hr, cv, ok in r['wins']:
            m = (bt >= a) & (bt < a + GATE_WIN); gate[m] = ok; whr[m] = hr
        df = pd.DataFrame(dict(beat=np.arange(1, len(bt) + 1), time_s=np.round(bt, 4), sample_index=r['idx'], detector_sample=dsamp,
                               detector_time_s=np.round(r['t'][dsamp], 4) if len(dsamp) else [], rr_s=np.round(rr, 4), hr_bpm=np.round(60 / rr, 2),
                               window_hr_bpm=np.round(whr, 2), window_gate_ok=gate))
        with open(p, 'w', newline='') as f:
            f.write(f'# source={os.path.basename(r["path"])}; column={r["col"]}; fs={r["fs"]:g}; detector=neotex_qrs_v2; params={json.dumps(self.params)}; '
                    f'snap_to_peak={self.v_snap.get()} (+-{int(SNAP_S * 1000)} ms, polarity {"+" if r["pol"] > 0 else "-"}); gate=HR {HR_MIN}-{HR_MAX}, RR-CV<{RRCV_MAX}, {GATE_WIN:g}s windows\n')
            df.to_csv(f, index=False)
        self.v_status.set(f'Exported {len(df):,} beats → {p}')

    def save_png(self):
        if not self.rec: return
        stem = os.path.splitext(os.path.basename(self.rec['path']))[0]
        p = filedialog.asksaveasfilename(title='Save view', defaultextension='.png', initialdir=os.path.dirname(self.rec['path']),
                                         initialfile=f'{stem}_t{self.cur:.0f}s.png', filetypes=[('PNG', '*.png')])
        if p: self.fig.savefig(p, dpi=200, facecolor=C['panel']); self.v_status.set(f'Saved {p}')

    def _tk_error(self, exc, val, tb):
        messagebox.showerror('Unexpected error', ''.join(traceback.format_exception(exc, val, tb))[-1500:])

def fmt(s):
    s = round(max(0.0, float(s)), 1); return f'{int(s // 60)}:{s % 60:04.1f}'

if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            import ctypes; ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception: pass
    Viewer(sys.argv[1] if len(sys.argv) > 1 else None).mainloop()
