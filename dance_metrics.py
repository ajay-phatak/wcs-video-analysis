#!/usr/bin/env python3
"""
Dance Metrics
=============
Computes analytical metrics from extracted pose data for West Coast Swing.

Four analysis categories:
  1. Leg Action       -- footwork, knee flex, rise/fall, step timing
  2. Body Action      -- shoulder/hip rotation, dissociation, sway, arm styling
  3. Weight & Counter -- connection point detection, lean, stretch/compression
  4. Musicality       -- beat detection, on-beat %, timing consistency, WCS pattern counts
"""

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
from scipy.signal import butter, correlate as _correlate, filtfilt, find_peaks, savgol_filter
from scipy.stats import pearsonr as _pearsonr

from pose_extraction import (
    KP, get_center, get_kp, body_height, torso_angle, shoulder_angle, hip_angle,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONF_MIN       = 0.25   # ignore keypoints below this confidence
SMOOTH_WINDOW  = 7      # Savitzky-Golay smoothing window (frames)
SMOOTH_POLY    = 2

# Connection point detection thresholds (fraction of body height)
HAND_HAND_FRAC = 0.25   # wrists within this fraction of body height → hand-to-hand
HAND_BACK_FRAC = 0.35   # wrist within this fraction of torso centre → hand-to-back


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _smooth(arr: np.ndarray) -> np.ndarray:
    if len(arr) < SMOOTH_WINDOW:
        return arr
    return savgol_filter(arr, SMOOTH_WINDOW, SMOOTH_POLY)


def _bandpass(sig: np.ndarray, fs: float, lo: float = 0.3, hi: float = 4.0) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass filter.
    Isolates dance-frequency motion (0.3–4 Hz by default), removing slow
    postural drift and high-frequency pose jitter before cross-correlation.
    Falls back to the original signal if filtering is not possible.
    """
    if len(sig) < 15 or fs <= 0:
        return sig
    nyq  = fs / 2.0
    lo_n = max(lo / nyq, 1e-4)
    hi_n = min(hi / nyq, 0.99)
    if lo_n >= hi_n:
        return sig
    try:
        b, a = butter(2, [lo_n, hi_n], btype="band")
        return filtfilt(b, a, sig)
    except Exception:
        return sig


def _detrend_fft(sig: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove a linear trend from sig, then return (fft_magnitudes, fft_freqs_placeholder).
    Returns the detrended signal so callers can pass it to np.fft.rfft.
    """
    if len(sig) < 3:
        return sig
    trend = np.polyval(np.polyfit(np.arange(len(sig)), sig, 1), np.arange(len(sig)))
    return sig - trend


def _valid(kps: np.ndarray, idx: int) -> bool:
    return kps.shape[0] > idx and kps[idx, 2] >= CONF_MIN


# ---------------------------------------------------------------------------
# Refined-pose extras: Halpe-26 foot keypoints (pass 2) and lifted 3D joints
# (pass 3). Both are optional — every consumer falls back to the COCO-17
# proxies when a pose file predates the refinement pipeline.
# ---------------------------------------------------------------------------

# Halpe-26 rows 17+ (rows 0-16 are COCO-17, same as KP)
FKP = {
    "left_big_toe": 20,   "right_big_toe": 21,
    "left_small_toe": 22, "right_small_toe": 23,
    "left_heel": 24,      "right_heel": 25,
}

# H36M-17 joint order of the lifted 3D poses (differs from COCO — see pose_lift.py)
KP3D = {
    "pelvis": 0, "right_hip": 1, "right_knee": 2, "right_ankle": 3,
    "left_hip": 4, "left_knee": 5, "left_ankle": 6, "spine": 7,
    "thorax": 8, "neck": 9, "head": 10, "left_shoulder": 11,
    "left_elbow": 12, "left_wrist": 13, "right_shoulder": 14,
    "right_elbow": 15, "right_wrist": 16,
}


def _feet_available(frames: list, dancer_id: int) -> bool:
    """True when the clip's poses carry usable Halpe-26 foot keypoints for this
    dancer (heels confident in a meaningful share of sampled frames)."""
    seen = hits = 0
    for f in frames[:: max(1, len(frames) // 100)]:
        kps = f["dancers"].get(dancer_id)
        if kps is None:
            continue
        seen += 1
        if _valid(kps, FKP["left_heel"]) or _valid(kps, FKP["right_heel"]):
            hits += 1
    return seen > 0 and hits / seen >= 0.5


def _kps3d(frame: dict, dancer_id: int) -> np.ndarray | None:
    """Lifted 3D joints (17,3) for a dancer in one frame, or None. JSON round-
    trips leave dancers3d with string keys and list values — normalise here."""
    d3 = frame.get("dancers3d")
    if not d3:
        return None
    v = d3.get(dancer_id, d3.get(str(dancer_id)))
    if v is None:
        return None
    return v if isinstance(v, np.ndarray) else np.asarray(v, dtype=float)


def _pose3d_available(frames: list, dancer_id: int) -> bool:
    for f in frames[:: max(1, len(frames) // 20)]:
        if _kps3d(f, dancer_id) is not None:
            return True
    return False


def _angle3d(k3: np.ndarray, a_idx: int, b_idx: int, c_idx: int) -> float | None:
    """Interior angle (degrees) at 3D joint b — rotation/camera invariant."""
    ba = k3[a_idx] - k3[b_idx]
    bc = k3[c_idx] - k3[b_idx]
    na, nc = float(np.linalg.norm(ba)), float(np.linalg.norm(bc))
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _times(frames: list, dancer_id: int) -> np.ndarray:
    return np.array([f["time_sec"] for f in frames if dancer_id in f["dancers"]])


def _kp_series(frames: list, dancer_id: int, kp_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, x_arr, y_arr) filtering by confidence."""
    times, xs, ys = [], [], []
    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        if _valid(kps, kp_idx):
            times.append(f["time_sec"])
            xs.append(kps[kp_idx, 0])
            ys.append(kps[kp_idx, 1])
    return np.array(times), np.array(xs), np.array(ys)


def _circular_range(angles_deg: np.ndarray) -> float:
    """
    Smallest arc (degrees) that contains all angles in the array.
    Avoids the 360° artifact that np.ptp produces near the ±180° wrap boundary.
    """
    if len(angles_deg) == 0:
        return 0.0
    a = np.sort(np.array(angles_deg) % 360)
    gaps = np.diff(np.concatenate([a, [a[0] + 360]]))
    return float(360.0 - np.max(gaps))


def _center_series(frames: list, dancer_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    times, xs, ys = [], [], []
    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        c = get_center(f["dancers"][dancer_id])
        if c[2] >= CONF_MIN:
            times.append(f["time_sec"])
            xs.append(c[0])
            ys.append(c[1])
    return np.array(times), np.array(xs), np.array(ys)


# ---------------------------------------------------------------------------
# Beat / audio extraction
# ---------------------------------------------------------------------------

def extract_audio_features(video_path: str) -> dict:
    """
    Extract a rich audio description of the song for music–movement matching.

    Beyond beats/tempo/energy, this characterises the song's *texture* over time
    (bouncy/percussive ↔ smooth/legato) and locates *accent moments* (hits, breaks,
    stabs) anywhere in the track — so we can later ask whether the dancer's movement
    quality matches the music, and whether they express the accents the song offers.

    Returns a dict:
        beat_times        np.ndarray  beat onsets (s)
        tempo             float       BPM
        rms_times         np.ndarray  ~100 ms grid (s)
        rms_vals          np.ndarray  RMS energy envelope
        onset_times       np.ndarray  onset-strength grid (s)
        onset_env         np.ndarray  onset strength envelope
        texture_times     np.ndarray  ~0.5 s grid (s)
        texture           np.ndarray  per-window bounciness in [0,1]
                                      (1 = punchy/percussive, 0 = smooth/legato)
        accent_times      np.ndarray  times of notable accents/hits/breaks (s)
        accent_strengths  np.ndarray  onset-strength prominence at each accent
        song_bounciness   float       mean texture (overall bouncy↔smooth, [0,1])
        song_dynamic_range float      RMS spread (p90−p10)/mean — accent dynamics
        accent_count      int
    """
    TEXTURE_HOP_S  = 0.5     # window for the bouncy↔smooth texture series
    TEMPO_MIN_BPM  = 65.0    # WCS-plausible tempo band. beat_track's dynamic-programming
    TEMPO_MAX_BPM  = 120.0   # tempo octave-jumps on syncopated songs (e.g. 172 or 48 for a
                             # true ~95), so we take the onset-autocorrelation peak inside
                             # this band, weighted by a prior, instead.
    TEMPO_PRIOR_BPM = 94.0   # centre of the log-normal tempo prior (typical WCS tempo).
    TEMPO_PRIOR_OCT = 0.4    # prior width in octaves — disambiguates harmonic peaks
                             # (e.g. 66 vs 95) without hard-excluding valid slow/fast tempos.

    audio_path = Path(tempfile.mktemp(suffix=".wav"))
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    try:
        subprocess.run(
            [ffmpeg_exe, "-y", "-i", video_path, "-ac", "1", "-ar", "22050",
             "-vn", str(audio_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        y, sr = librosa.load(str(audio_path), sr=22050)

        # ── beats / tempo ──────────────────────────────────────────────────
        # beat_track's dynamic-programming tempo estimate octave-jumps on
        # syncopated / WCS songs (e.g. reads 172 or 48 for a true ~95). Take the
        # strongest periodicity of the onset envelope within a WCS-plausible BPM
        # band instead, then force the beat grid to that tempo.
        dur_s     = len(y) / sr if sr else 0.0
        tempo_hop = 512
        oenv_t    = librosa.onset.onset_strength(y=y, sr=sr, hop_length=tempo_hop)
        ac        = librosa.autocorrelate(oenv_t)
        tfreq     = librosa.tempo_frequencies(len(ac), sr=sr, hop_length=tempo_hop)
        band      = (tfreq >= TEMPO_MIN_BPM) & (tfreq <= TEMPO_MAX_BPM)
        if band.any() and float(np.max(ac[band])) > 0:
            # Weight the in-band autocorrelation by a log-normal prior centred on a
            # typical WCS tempo to disambiguate harmonic peaks (e.g. 66 vs 95 vs 172),
            # which sit within a few % of each other in the raw autocorrelation.
            prior = np.exp(-0.5 * ((np.log2(np.maximum(tfreq, 1e-6)) -
                                    np.log2(TEMPO_PRIOR_BPM)) / TEMPO_PRIOR_OCT) ** 2)
            score = np.where(band, ac * prior, -np.inf)
            tempo_val = float(tfreq[int(np.argmax(score))])
        else:
            _t, _ = librosa.beat.beat_track(y=y, sr=sr, units="frames")
            tempo_val = float(np.atleast_1d(_t)[0])

        # Beat grid forced to the resolved tempo (phase-aligned to onsets); fall
        # back to a uniform grid if this librosa lacks the bpm= override.
        try:
            _, beat_frames = librosa.beat.beat_track(
                y=y, sr=sr, bpm=tempo_val, units="frames")
            beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        except TypeError:
            beat_times = np.arange(0.0, dur_s, 60.0 / max(tempo_val, 1e-6))
        beat_dur  = 60.0 / max(tempo_val, 60.0)

        # ── RMS energy envelope (100 ms) ───────────────────────────────────
        rms_hop  = int(sr * 0.1)
        rms      = librosa.feature.rms(y=y, hop_length=rms_hop)[0]
        rms_t    = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=rms_hop)

        # ── onset strength envelope (≈23 ms) ───────────────────────────────
        onset_hop = 512
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=onset_hop)
        onset_t   = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr,
                                           hop_length=onset_hop)
        onset_dt  = float(np.mean(np.diff(onset_t))) if len(onset_t) > 1 else 0.023

        # ── harmonic / percussive split → per-window percussive ratio ──────
        # Percussive energy = punchy/bouncy texture; harmonic = sustained/legato.
        try:
            y_harm, y_perc = librosa.effects.hpss(y)
            perc_env = librosa.feature.rms(y=y_perc, hop_length=onset_hop)[0]
            harm_env = librosa.feature.rms(y=y_harm, hop_length=onset_hop)[0]
            n = min(len(perc_env), len(harm_env), len(onset_env))
            perc_ratio_env = perc_env[:n] / (perc_env[:n] + harm_env[:n] + 1e-9)
        except Exception:
            n = len(onset_env)
            perc_ratio_env = np.full(n, 0.5)

        # ── bouncy↔smooth texture series on a ~0.5 s grid ──────────────────
        # Combine percussive ratio (timbre) with onset density (attack rate);
        # both rise in punchy/staccato passages and fall in smooth/legato ones.
        onset_norm = onset_env[:n] / (np.percentile(onset_env[:n], 95) + 1e-9)
        onset_norm = np.clip(onset_norm, 0.0, 1.0)
        dur        = float(onset_t[n - 1]) if n > 0 else 0.0
        n_win      = max(1, int(dur / TEXTURE_HOP_S) + 1)
        texture_t  = np.arange(n_win) * TEXTURE_HOP_S
        texture    = np.zeros(n_win)
        for i, wt in enumerate(texture_t):
            mask = (onset_t[:n] >= wt) & (onset_t[:n] < wt + TEXTURE_HOP_S)
            if mask.any():
                texture[i] = 0.6 * float(np.mean(perc_ratio_env[mask])) \
                             + 0.4 * float(np.mean(onset_norm[mask]))
        texture = np.clip(texture, 0.0, 1.0)

        # ── accent events: onset peaks that stand out from a local baseline ─
        accent_times     = np.array([])
        accent_strengths = np.array([])
        if len(onset_env) > 5:
            med = float(np.median(onset_env))
            mad = float(np.median(np.abs(onset_env - med))) + 1e-9
            min_gap = max(1, int(round((beat_dur * 0.5) / max(onset_dt, 1e-6))))
            peaks, props = find_peaks(onset_env, prominence=3.0 * mad, distance=min_gap)
            if len(peaks):
                proms = props["prominences"]
                # Keep the more *notable* accents (hits/breaks), not every beat.
                keep = proms >= np.percentile(proms, 70)
                accent_times     = onset_t[peaks][keep]
                accent_strengths = proms[keep]

        # ── song-character summary scalars ─────────────────────────────────
        song_bounciness = float(np.mean(texture)) if len(texture) else 0.0
        if len(rms):
            rng = float(np.percentile(rms, 90) - np.percentile(rms, 10))
            song_dynamic_range = round(rng / (float(np.mean(rms)) + 1e-9), 3)
        else:
            song_dynamic_range = 0.0

        return {
            "beat_times":         beat_times,
            "tempo":              tempo_val,
            "rms_times":          rms_t,
            "rms_vals":           rms,
            "onset_times":        onset_t,
            "onset_env":          onset_env,
            "texture_times":      texture_t,
            "texture":            texture,
            "accent_times":       accent_times,
            "accent_strengths":   accent_strengths,
            "song_bounciness":    round(song_bounciness, 3),
            "song_dynamic_range": song_dynamic_range,
            "accent_count":       int(len(accent_times)),
        }
    finally:
        if audio_path.exists():
            audio_path.unlink()


def extract_beats(video_path: str) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Backward-compatible wrapper: (beat_times, tempo, rms_times, rms_vals)."""
    a = extract_audio_features(video_path)
    return a["beat_times"], a["tempo"], a["rms_times"], a["rms_vals"]


# ---------------------------------------------------------------------------
# Step / weight-transfer detection
# ---------------------------------------------------------------------------

def detect_step_events(frames: list, dancer_id: int, fps: float) -> dict:
    """
    Detect all weight transfers via hip lateral oscillation, then classify each
    as 'articulated' (ankle lifts clear of the ground) or 'weight_only' (in-place
    weight shift with no heel lift — triple-step & counts, anchors).

    Classification: measure the maximum upward ankle excursion (decrease in y
    in image coords, since y increases downward) in a window around each event,
    normalised to body height.  If either ankle rises more than LIFT_THRESHOLD
    it is articulated — regardless of whether the foot travels sideways.

    Also computes per-frame balance state: 'one_foot' when one ankle is clearly
    elevated relative to the other (≥ BALANCE_THRESH * body_height), 'two_foot'
    when both ankles are at similar height.

    Returns:
        {
            "articulated":  [float, ...]  # times (s) — heel clearly lifts
            "weight_only":  [float, ...]  # times (s) — in-place weight shift
            "all":          [float, ...]  # combined, sorted
            "one_foot_pct": float         # % of frames in 1-foot balance
            "two_foot_pct": float         # % of frames in 2-foot balance
        }
    """
    MIN_STEP_GAP     = 0.12   # minimum seconds between consecutive events
    LIFT_THRESHOLD   = 0.05   # ankle must rise > 5% of body height to be articulated
    LIFT_WINDOW      = 0.25   # seconds around event to search for heel lift
    BALANCE_THRESH   = 0.05   # ankle height difference > 5% bh → 1-foot balance
    TRAVEL_THRESHOLD = 0.05   # stance-center shift across an event > 5% bh → traveling
    #   Stance center = mean(left_ankle_x, right_ankle_x). Unlike hip center, stance
    #   center doesn't move when the dancer rocks weight across planted feet — it only
    #   moves when a foot lands in a new location, so it isolates "did a foot move at
    #   this event?" from "did body weight shift?". The classification uses the
    #   displacement of stance_center in a window BEFORE vs AFTER the event (not
    #   between consecutive events), so each event is judged on its own step motion
    #   rather than the cumulative travel from prior steps.
    TRAVEL_WIN_INNER = 0.05   # seconds: skip window right at the event (mid-transition)
    TRAVEL_WIN_OUTER = 0.20   # seconds: averaging window extends to ± this from event
    #   5% bh ≈ 8 cm: catches heel raises, brush-throughs, and toe-touch weight shifts
    #   as well as foot fully off the ground; preserves clear pro vs amateur differential

    # ---- body height for normalisation ----
    bh_vals = [body_height(f["dancers"][dancer_id])
               for f in frames
               if dancer_id in f["dancers"] and body_height(f["dancers"][dancer_id]) > 10]
    bh_mean = float(np.mean(bh_vals)) if bh_vals else 100.0

    # ---- detect all weight-change events via hip oscillation ----
    t_c, x_c, _ = _center_series(frames, dancer_id)
    all_times: list[float] = []

    if len(t_c) >= SMOOTH_WINDOW + 2:
        # Scale prominence to body height so it works across video resolutions
        prominence = max(2.0, bh_mean * 0.02)
        x_sm       = _smooth(x_c)
        gap_fr     = max(3, int(len(t_c) * MIN_STEP_GAP / (t_c[-1] - t_c[0] + 1e-6)))
        peaks,   _ = find_peaks( x_sm, distance=gap_fr, prominence=prominence)
        valleys, _ = find_peaks(-x_sm, distance=gap_fr, prominence=prominence)
        all_times  = sorted(float(t_c[i]) for i in list(peaks) + list(valleys))

    if len(all_times) <= 5:
        # Fallback: ankle vertical peaks
        for kp_idx in [KP["left_ankle"], KP["right_ankle"]]:
            t_a, _, y_a = _kp_series(frames, dancer_id, kp_idx)
            if len(y_a) < SMOOTH_WINDOW + 2:
                continue
            y_sm   = _smooth(y_a)
            gap_fr = max(3, int(len(y_a) * MIN_STEP_GAP / (t_a[-1] - t_a[0] + 1e-6)))
            pk, _  = find_peaks(y_sm, prominence=bh_mean * 0.02, distance=gap_fr)
            all_times.extend(float(t_a[i]) for i in pk)
        all_times.sort()

    if not all_times:
        return {"articulated": [], "weight_only": [], "all": [],
                "articulated_traveling": [], "articulated_in_place": [],
                "weight_only_traveling": [], "weight_only_in_place": [],
                "one_foot_pct": 0.0, "two_foot_pct": 0.0,
                "one_foot_airborne_pct": 0.0, "ball_foot_pct": 0.0,
                "foot_kps_used": False}

    # ---- build smoothed vertical series for lift measurement ----
    # With Halpe-26 feet, the HEEL is the true lift signal ("heel lifts clear of
    # the ground" is the definition of an articulated step); the ankle sits a
    # variable distance above ground depending on camera angle, so it's only the
    # fallback for pass-1 poses.
    feet_ok = _feet_available(frames, dancer_id)
    lift_kp_idxs = ([FKP["left_heel"], FKP["right_heel"]] if feet_ok
                    else [KP["left_ankle"], KP["right_ankle"]])
    ankle_series: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for kp_idx in lift_kp_idxs:
        t_a, _, y_a = _kp_series(frames, dancer_id, kp_idx)
        if len(t_a) >= SMOOTH_WINDOW + 2:
            ankle_series[kp_idx] = (t_a, _smooth(y_a))

    # ---- classify each event by heel lift (upward ankle excursion) ----
    articulated: list[float] = []
    weight_only: list[float] = []

    for ev_t in all_times:
        max_lift_norm = 0.0
        for kp_idx, (t_a, y_sm) in ankle_series.items():
            mask = (t_a >= ev_t - LIFT_WINDOW) & (t_a <= ev_t + LIFT_WINDOW * 0.5)
            if mask.sum() < 3:
                continue
            y_win = y_sm[mask]
            # y increases downward in image coords, so a heel lift = y decreasing.
            # Upward excursion = 75th-percentile y (near-ground baseline) minus minimum y.
            baseline_y = float(np.percentile(y_win, 75))
            min_y      = float(np.min(y_win))
            lift       = (baseline_y - min_y) / bh_mean
            max_lift_norm = max(max_lift_norm, lift)

        if max_lift_norm >= LIFT_THRESHOLD:
            articulated.append(ev_t)
        else:
            weight_only.append(ev_t)

    # ---- traveling vs in-place: stance-center displacement between events ----
    # Build per-frame stance_center = mean of both ankle x's (or the confident one).
    sc_t: list[float] = []
    sc_x: list[float] = []
    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        la = kps[KP["left_ankle"]]
        ra = kps[KP["right_ankle"]]
        xs = []
        if la[2] >= CONF_MIN:
            xs.append(float(la[0]))
        if ra[2] >= CONF_MIN:
            xs.append(float(ra[0]))
        if xs:
            sc_t.append(f["time_sec"])
            sc_x.append(float(np.mean(xs)))

    # Smooth with ~0.4 s uniform window to suppress foot-in-air transients
    # without blurring genuine travel between closely-spaced events.
    articulated_traveling: list[float] = []
    articulated_in_place:  list[float] = []
    weight_only_traveling: list[float] = []
    weight_only_in_place:  list[float] = []

    if len(sc_t) >= SMOOTH_WINDOW + 2:
        sc_t_arr = np.array(sc_t)
        sc_x_arr = np.array(sc_x)
        dt_sc = float(np.mean(np.diff(sc_t_arr))) if len(sc_t_arr) > 1 else 1.0 / fps
        win = max(3, int(round(0.4 / max(dt_sc, 1e-6))))
        if win % 2 == 0:
            win += 1
        # Uniform moving average via convolution (same length as input)
        kernel = np.ones(win) / win
        sc_sm = np.convolve(sc_x_arr, kernel, mode="same")

        art_set = set(round(t, 4) for t in articulated)

        for ev_t in all_times:
            # Measure stance-center displacement across THIS event (pre vs post
            # window), not relative to the previous event — otherwise we'd be
            # measuring the PRIOR step's travel at each event.
            mask_pre  = (sc_t_arr >= ev_t - TRAVEL_WIN_OUTER) & \
                        (sc_t_arr <= ev_t - TRAVEL_WIN_INNER)
            mask_post = (sc_t_arr >= ev_t + TRAVEL_WIN_INNER) & \
                        (sc_t_arr <= ev_t + TRAVEL_WIN_OUTER)
            if mask_pre.sum() < 2 or mask_post.sum() < 2:
                # Event too close to video edge — default to in-place
                is_traveling = False
            else:
                pre_sc  = float(np.mean(sc_sm[mask_pre]))
                post_sc = float(np.mean(sc_sm[mask_post]))
                delta = abs(post_sc - pre_sc) / bh_mean
                is_traveling = delta > TRAVEL_THRESHOLD

            is_articulated = round(ev_t, 4) in art_set
            if is_articulated and is_traveling:
                articulated_traveling.append(ev_t)
            elif is_articulated:
                articulated_in_place.append(ev_t)
            elif is_traveling:
                weight_only_traveling.append(ev_t)
            else:
                weight_only_in_place.append(ev_t)
    else:
        # No stance-center signal — default everything to in-place.
        articulated_in_place = list(articulated)
        weight_only_in_place = list(weight_only)

    # ---- per-frame 1-foot vs 2-foot balance ----
    # Ankle proxy (all pose files): one ankle elevated ≥5% bh → "1-foot", which
    # deliberately lumps heel raises / brushes / toe-touches in with a fully
    # lifted foot (an ankle can't tell them apart).
    # With Halpe-26 feet two finer states are also measured:
    #   airborne = one foot's LOWEST contact point (heel/toes) clearly above the
    #              grounded foot → true single-leg balance, no floor contact.
    #   ball     = a heel raised while its own toes stay grounded → dancing on
    #              the ball of the foot (rolling action), still floor contact.
    one_foot_count = 0
    two_foot_count = 0
    airborne_count = 0
    ball_count     = 0

    def _foot_pts(kps, side):
        """(lowest_contact_y, heel_y, toe_y) for one foot — None where unmeasurable."""
        heel = kps[FKP[f"{side}_heel"]]     if _valid(kps, FKP[f"{side}_heel"])     else None
        big  = kps[FKP[f"{side}_big_toe"]]  if _valid(kps, FKP[f"{side}_big_toe"])  else None
        small = kps[FKP[f"{side}_small_toe"]] if _valid(kps, FKP[f"{side}_small_toe"]) else None
        toe_y = max((float(p[1]) for p in (big, small) if p is not None), default=None)
        ys = [float(p[1]) for p in (heel, big, small) if p is not None]
        low_y = max(ys) if ys else None          # image y grows downward → max = lowest
        heel_y = float(heel[1]) if heel is not None else None
        return low_y, heel_y, toe_y

    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        bh  = body_height(kps)
        if bh < 10:
            continue
        la = kps[KP["left_ankle"]]
        ra = kps[KP["right_ankle"]]
        if la[2] < CONF_MIN or ra[2] < CONF_MIN:
            continue
        # One foot elevated → larger y difference (y↑ = foot off ground)
        diff = abs(float(la[1]) - float(ra[1])) / bh
        if diff > BALANCE_THRESH:
            one_foot_count += 1
        else:
            two_foot_count += 1

        if feet_ok:
            l_low, l_heel, l_toe = _foot_pts(kps, "left")
            r_low, r_heel, r_toe = _foot_pts(kps, "right")
            if l_low is not None and r_low is not None:
                floor_y = max(l_low, r_low)      # the grounded foot defines the floor
                if abs(l_low - r_low) / bh > BALANCE_THRESH:
                    airborne_count += 1
                for heel_y, toe_y, low in ((l_heel, l_toe, l_low), (r_heel, r_toe, r_low)):
                    # ball-of-foot: heel clearly above the floor line, toes on it
                    if (heel_y is not None and toe_y is not None
                            and (floor_y - heel_y) / bh > BALANCE_THRESH
                            and (floor_y - toe_y) / bh <= BALANCE_THRESH):
                        ball_count += 1
                        break

    total_balance = one_foot_count + two_foot_count
    one_foot_pct = round(100.0 * one_foot_count / max(total_balance, 1), 1)
    two_foot_pct = round(100.0 * two_foot_count / max(total_balance, 1), 1)
    one_foot_airborne_pct = round(100.0 * airborne_count / max(total_balance, 1), 1) if feet_ok else 0.0
    ball_foot_pct         = round(100.0 * ball_count     / max(total_balance, 1), 1) if feet_ok else 0.0

    return {
        "articulated":            articulated,
        "weight_only":            weight_only,
        "all":                    all_times,
        "articulated_traveling":  articulated_traveling,
        "articulated_in_place":   articulated_in_place,
        "weight_only_traveling":  weight_only_traveling,
        "weight_only_in_place":   weight_only_in_place,
        "one_foot_pct":           one_foot_pct,
        "two_foot_pct":           two_foot_pct,
        "one_foot_airborne_pct":  one_foot_airborne_pct,
        "ball_foot_pct":          ball_foot_pct,
        "foot_kps_used":          feet_ok,
    }


# ---------------------------------------------------------------------------
# Articulation quality (knee/hip joint angles on articulated steps)
# ---------------------------------------------------------------------------

def _angle_at(kps: np.ndarray, a_idx: int, b_idx: int, c_idx: int) -> float | None:
    """Interior angle (degrees) at joint b, between vectors b→a and b→c.

    Returns None if any of the three keypoints is below CONF_MIN or degenerate.
    For a leg, angle_at(hip, knee, ankle) is the knee angle (180° = straight leg).
    """
    if kps[a_idx, 2] < CONF_MIN or kps[b_idx, 2] < CONF_MIN or kps[c_idx, 2] < CONF_MIN:
        return None
    ba = kps[a_idx, :2] - kps[b_idx, :2]
    bc = kps[c_idx, :2] - kps[b_idx, :2]
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na < 1e-6 or nc < 1e-6:
        return None
    cos = float(np.dot(ba, bc) / (na * nc))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


_ART_QUALITY_KEYS = (
    "art_free_knee_flex_deg", "art_free_hip_flex_deg", "art_weighted_knee_flex_deg",
    "art_free_knee_p90", "art_weighted_knee_p90",
    "art_knee_hip_coord", "art_smoothness", "art_straighten_pct", "art_prep_pct",
    "art_ankle_lift",
    "art_toe_first_pct", "art_heel_first_pct", "art_flat_pct", "art_ball_only_pct",
    "art_roll_lag_ms", "art_roll_n",
)


def _articulation_per_step(frames: list, dancer_id: int, fps: float,
                           step_data: dict, bh_mean: float) -> list:
    """
    Per-articulated-step records of leg articulation, separating the FREE (moving)
    leg from the STANDING (weighted) leg — the two have different jobs and must not
    be conflated. Shared core for both the medians (_articulation_quality) and the
    distribution / accent-timing / coupling detail (compute_articulation_detail).

    Generic-step model (the user's, with musical exceptions like a deep accent lunge):
    the *moving* leg bends while FREE to prepare/place the foot → (body flight if
    traveling) → straightens as weight ARRIVES on it. The *standing* leg bears weight
    and may sink/load. Free-leg prep flexion does NOT lower the body (weight is on the
    other leg); weighted-leg flexion is what actually "gets lower". The moving leg is
    the one whose ankle travels most vertically; its FREE phase is when its own ankle
    is raised above its resting level (robust to the far leg being occluded side-on).

    Returns one dict per usable articulated step, with keys (nan where unmeasurable):
        t, free_knee, free_hip, wt_knee, coord, smooth, straighten, prep,
        ankle_lift, pitch_range, com_drop
    """
    art_times = step_data.get("articulated", [])
    if not art_times:
        return []

    legs = {
        "left":  (KP["left_hip"],  KP["left_knee"],  KP["left_ankle"],  KP["left_shoulder"]),
        "right": (KP["right_hip"], KP["right_knee"], KP["right_ankle"], KP["right_shoulder"]),
    }
    # H36M indices for the 3D angle upgrade (hip, knee, ankle, thorax)
    legs3d = {
        "left":  (KP3D["left_hip"],  KP3D["left_knee"],  KP3D["left_ankle"],  KP3D["thorax"]),
        "right": (KP3D["right_hip"], KP3D["right_knee"], KP3D["right_ankle"], KP3D["thorax"]),
    }
    # Joint angles: lifted 3D when available (rotation/camera-invariant — a knee
    # bend reads the same regardless of where the camera stood, so you-vs-pro
    # comparisons aren't biased by differing camera angles), else 2D projection.
    # Foot-free detection: the big toe directly encodes ground contact — a foot
    # on the ball (heel up, toes down) is WEIGHTED, which the ankle proxy
    # misreads as free. Toe used when Halpe-26 feet are present.
    use_3d  = _pose3d_available(frames, dancer_id)
    feet_ok = _feet_available(frames, dancer_id)
    toe_idx = {"left": FKP["left_big_toe"], "right": FKP["right_big_toe"]}

    heel_idx = {"left": FKP["left_heel"], "right": FKP["right_heel"]}

    series: dict = {leg: {"t": [], "knee": [], "hip": [], "ankle_y": [], "heel_y": []}
                    for leg in legs}
    b_t, b_com, b_pit = [], [], []      # body-level: time, hip-centre y, torso pitch
    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        k3  = _kps3d(f, dancer_id) if use_3d else None
        for leg, (hip_i, knee_i, ank_i, sh_i) in legs.items():
            if k3 is not None:
                h3, k3i, a3, th3 = legs3d[leg]
                knee_a = _angle3d(k3, h3, k3i, a3)
                hip_a  = _angle3d(k3, th3, h3, k3i)
            else:
                knee_a = _angle_at(kps, hip_i, knee_i, ank_i)
                hip_a  = _angle_at(kps, sh_i, hip_i, knee_i)
            if knee_a is None or hip_a is None:
                continue
            # vertical track of the foot: big toe (true ground contact) if
            # refined, else ankle
            foot_i = toe_idx[leg] if feet_ok else ank_i
            if not _valid(kps, foot_i):
                foot_i = ank_i
            s = series[leg]
            s["t"].append(f["time_sec"])
            s["knee"].append(knee_a)
            s["hip"].append(hip_a)
            s["ankle_y"].append(float(kps[foot_i, 1]) if _valid(kps, foot_i) else np.nan)
            h_i = heel_idx[leg]
            s["heel_y"].append(float(kps[h_i, 1]) if (feet_ok and _valid(kps, h_i)) else np.nan)
        c = get_center(kps)
        if c[2] >= CONF_MIN:
            b_t.append(f["time_sec"])
            b_com.append(float(c[1]))
            b_pit.append(torso_angle(kps) if kps[KP["head"], 2] >= CONF_MIN else np.nan)

    for leg in series:
        s = series[leg]
        s["t"]       = np.array(s["t"])
        s["knee"]    = _smooth(np.array(s["knee"])) if len(s["knee"]) >= SMOOTH_WINDOW else np.array(s["knee"])
        s["hip"]     = _smooth(np.array(s["hip"]))  if len(s["hip"])  >= SMOOTH_WINDOW else np.array(s["hip"])
        s["ankle_y"] = np.array(s["ankle_y"])
        s["heel_y"]  = np.array(s["heel_y"])
    b_t   = np.array(b_t)
    b_com = np.array(b_com)
    b_pit = np.array(b_pit)

    W        = 0.45                       # seconds around each step event
    LIFT_TOL = 0.05 * max(bh_mean, 1.0)   # ankle raised this far above rest = free

    def _window(leg, t_ev):
        s = series[leg]
        if len(s["t"]) < 4:
            return None
        m = (s["t"] >= t_ev - W) & (s["t"] <= t_ev + W)
        if m.sum() < 5:
            return None
        return s["t"][m], s["knee"][m], s["hip"][m], s["ankle_y"][m], s["heel_y"][m]

    records: list = []
    for t_ev in art_times:
        win = {leg: _window(leg, t_ev) for leg in legs}
        win = {leg: w for leg, w in win.items() if w is not None}
        if not win:
            continue

        def _ankle_exc(w):
            ay = w[3][~np.isnan(w[3])]
            return float(np.max(ay) - np.min(ay)) if len(ay) >= 3 else -1.0
        move_leg  = max(win, key=lambda lg: _ankle_exc(win[lg]))
        tt, kn, hp, an, he = win[move_leg]
        stand_leg = next((lg for lg in legs if lg != move_leg and lg in win), None)

        straight_ref = float(np.max(kn))
        knee_min     = float(np.min(kn))
        depth        = straight_ref - knee_min
        if depth < 2.0:        # negligible bend — not a genuine articulation; skip
            continue

        # FREE phase of the moving foot: its ankle raised above its own resting level.
        _an_ok   = an[~np.isnan(an)]
        an_med   = float(np.median(_an_ok)) if len(_an_ok) else np.nan
        free_msk = (~np.isnan(an)) & (an < an_med - LIFT_TOL)   # smaller y = higher foot
        if free_msk.sum() >= 3:
            kn_free, hp_free = kn[free_msk], hp[free_msk]
        else:                                                   # fallback: pre-event prep
            pre = tt <= t_ev
            kn_free = kn[pre] if pre.sum() >= 3 else kn
            hp_free = hp[pre] if pre.sum() >= 3 else hp

        rec: dict = {
            "t":         float(t_ev),
            "free_knee": float(np.max(kn_free) - np.min(kn_free)),
            "free_hip":  float(np.max(hp_free) - np.min(hp_free)),
            "wt_knee":   np.nan, "coord": np.nan, "smooth": np.nan,
            "straighten": np.nan, "prep": 0.0,
            "ankle_lift": np.nan, "pitch_range": np.nan, "com_drop": np.nan,
        }

        # standing-leg knee flexion (sink/load while weighted)
        if stand_leg is not None:
            kn_s = win[stand_leg][1]
            rec["wt_knee"] = float(np.max(kn_s) - np.min(kn_s))

        # proportionality of the gather: free-leg knee vs hip flex/extend together
        if len(kn_free) >= 4 and np.std(kn_free) > 1e-6 and np.std(hp_free) > 1e-6:
            rec["coord"] = float(_pearsonr(kn_free, hp_free)[0])

        # smoothness of the moving leg: monotonic down to deepest bend then up.
        i_flex_w = int(np.argmin(kn))
        TOL  = 1.0   # degrees: deadband below the smoothed-pose noise floor
        desc = np.diff(kn[:i_flex_w + 1])
        asc  = np.diff(kn[i_flex_w:])
        parts = []
        if len(desc):
            parts.append((float(np.mean(desc <= TOL)),  len(desc)))
        if len(asc):
            parts.append((float(np.mean(asc >= -TOL)), len(asc)))
        if parts:
            wsum = sum(w for _, w in parts)
            rec["smooth"] = sum(v * w for v, w in parts) / wsum

        # straighten recovery: how much of the moving-leg bend re-extends by window end
        knee_end = float(np.mean(kn[-3:])) if len(kn) >= 3 else float(kn[-1])
        rec["straighten"] = 100.0 * max(0.0, knee_end - knee_min) / (depth + 1e-6)

        # prep sequencing: deepest bend while the foot is FREE, then it straightens after
        i_min       = int(np.argmin(kn))
        bend_free   = bool(free_msk[i_min]) if free_msk.sum() >= 3 else (tt[i_min] <= t_ev + 0.05)
        post        = kn[tt >= float(tt[i_min])]
        extends_aft = len(post) >= 2 and float(post[-1]) > knee_min + 0.25 * depth
        rec["prep"] = 1.0 if (bend_free and extends_aft) else 0.0

        # ankle lift proxy of the MOVING foot
        an_valid = an[~np.isnan(an)]
        if len(an_valid) >= 3:
            rec["ankle_lift"] = float(np.max(an_valid) - np.min(an_valid)) / max(bh_mean, 1.0)

        # Landing roll-through: which part of the moving foot takes weight first.
        # After the lift apex, find when the toe and the heel each return to their
        # grounded level; the sign of the lag classifies the landing:
        #   toe first (rolls ball→heel), heel first, flat (same frame at ~30 fps),
        #   or ball-only (heel never grounds in the window — triples/anchors held
        #   on the ball). Needs real toe+heel tracks, i.e. Halpe-26 feet.
        rec["toe_heel_lag_ms"] = np.nan
        rec["landing"] = "na"
        if feet_ok:
            toe_v, heel_v = ~np.isnan(an), ~np.isnan(he)
            if toe_v.sum() >= 5 and heel_v.sum() >= 5:
                g_toe  = float(np.percentile(an[toe_v], 85))   # grounded baseline
                g_heel = float(np.percentile(he[heel_v], 85))  # (foot is down most of the window)
                tol    = 0.02 * max(bh_mean, 1.0)
                i_ap   = int(np.nanargmin(an))                 # lift apex (highest toe)

                def _contact(y, valid, ground):
                    for i in range(i_ap, len(y)):
                        if valid[i] and y[i] >= ground - tol:
                            return float(tt[i])
                    return None

                toe_c  = _contact(an, toe_v, g_toe)
                heel_c = _contact(he, heel_v, g_heel)
                if toe_c is not None and heel_c is not None:
                    lag = (heel_c - toe_c) * 1000.0    # + = toe grounded first
                    rec["toe_heel_lag_ms"] = float(lag)
                    rec["landing"] = "toe" if lag > 15 else ("heel" if lag < -15 else "flat")
                elif toe_c is not None:
                    rec["landing"] = "ball"

        # body channels in the window: torso pitch range (deg) and COM vertical drop (BH)
        if len(b_t):
            bm = (b_t >= t_ev - W) & (b_t <= t_ev + W)
            if bm.sum() >= 3:
                com = b_com[bm]
                rec["com_drop"] = float(np.max(com) - np.min(com)) / max(bh_mean, 1.0)
                pit = b_pit[bm]
                pit = pit[~np.isnan(pit)]
                if len(pit) >= 3:
                    rec["pitch_range"] = float(np.max(pit) - np.min(pit))

        records.append(rec)

    return records


def _articulation_quality(frames: list, dancer_id: int, fps: float,
                          step_data: dict, bh_mean: float) -> dict:
    """Median (+ p90 ceiling) articulation-quality metrics over articulated steps.

    See _articulation_per_step for the model. Medians are robust to accent-lunge
    outliers; the p90 'ceiling' keys expose how deep the dancer can go on their
    biggest steps (the dynamic range), which the median hides.
    """
    recs = _articulation_per_step(frames, dancer_id, fps, step_data, bh_mean)
    angle_source = "3d" if _pose3d_available(frames, dancer_id) else "2d"
    zero = {"art_step_count": 0, "art_angle_source": angle_source,
            **{k: 0.0 for k in _ART_QUALITY_KEYS}}
    if not recs:
        return zero

    def _col(key):
        return [r[key] for r in recs if not (isinstance(r[key], float) and np.isnan(r[key]))]

    def _md(x, nd=1):
        return round(float(np.median(x)), nd) if x else 0.0

    def _p90(x):
        return round(float(np.percentile(x, 90)), 1) if x else 0.0

    fk, wk = _col("free_knee"), _col("wt_knee")
    prep = [r["prep"] for r in recs]

    # landing roll-through breakdown (measured landings only)
    landings = [r.get("landing", "na") for r in recs]
    n_land = sum(1 for l in landings if l != "na")
    def _lpct(kind):
        return round(100.0 * landings.count(kind) / n_land, 1) if n_land else 0.0
    # mean, not median: lags quantize to frame steps (±33 ms at 30 fps) and the
    # dominant same-frame "flat" bucket pins the median to 0; the mean keeps the
    # toe-first vs heel-first skew visible.
    lags = _col("toe_heel_lag_ms")
    roll_lag = round(float(np.mean(lags)), 1) if lags else 0.0

    return {
        "art_step_count":             len(recs),
        "art_angle_source":           angle_source,
        "art_toe_first_pct":          _lpct("toe"),
        "art_heel_first_pct":         _lpct("heel"),
        "art_flat_pct":               _lpct("flat"),
        "art_ball_only_pct":          _lpct("ball"),
        "art_roll_lag_ms":            roll_lag,
        "art_roll_n":                 n_land,
        "art_free_knee_flex_deg":     _md(fk),
        "art_free_hip_flex_deg":      _md(_col("free_hip")),
        "art_weighted_knee_flex_deg": _md(wk),
        "art_free_knee_p90":          _p90(fk),
        "art_weighted_knee_p90":      _p90(wk),
        "art_knee_hip_coord":         _md(_col("coord"), 3),
        "art_smoothness":             _md(_col("smooth"), 3),
        "art_straighten_pct":         _md(_col("straighten")),
        "art_prep_pct":               round(100.0 * float(np.mean(prep)), 1) if prep else 0.0,
        "art_ankle_lift":             _md(_col("ankle_lift"), 3),
    }


def compute_articulation_detail(frames: list, dancer_id: int, fps: float,
                                step_data: dict, accent_times: np.ndarray,
                                beat_dur: float, bh_mean: float | None = None) -> dict:
    """
    Deep per-step detail for movement quality & musicality (always-on):

      • DISTRIBUTIONS — p50/p75/p90/max of free-leg prep flex, standing-leg flex, and
        ankle lift, so a low ceiling (compressed range) is distinguishable from a steady
        gap. "Do they go for it on some steps, or is every step the same?"
      • ACCENT vs ANCHOR — mean bend depth on steps that land on a musical accent vs
        steps that don't (accent = within half a beat of a detected accent). "Does the
        dynamic movement concentrate on the music's moments?"
      • COUPLING — how step amplitude is regulated: foot-lift↔free-leg-flex,
        body-pitch↔free-leg-flex, standing-leg-sink↔COM-drop (Pearson r per dancer).
    """
    if bh_mean is None:
        bh_vals = [body_height(f["dancers"][dancer_id])
                   for f in frames if dancer_id in f["dancers"]
                   and body_height(f["dancers"][dancer_id]) > 10]
        bh_mean = float(np.mean(bh_vals)) if bh_vals else 1.0

    recs = _articulation_per_step(frames, dancer_id, fps, step_data, bh_mean)
    if not recs:
        return {"step_count": 0}

    def _arr(key):
        return np.array([r[key] for r in recs], dtype=float)

    def _dist(a):
        a = a[~np.isnan(a)]
        if len(a) == 0:
            return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "max": 0.0}
        return {"p50": round(float(np.percentile(a, 50)), 1),
                "p75": round(float(np.percentile(a, 75)), 1),
                "p90": round(float(np.percentile(a, 90)), 1),
                "max": round(float(np.max(a)), 1)}

    def _corr(a, b):
        m = ~(np.isnan(a) | np.isnan(b))
        if m.sum() < 8 or np.std(a[m]) < 1e-9 or np.std(b[m]) < 1e-9:
            return None
        return round(float(_pearsonr(a[m], b[m])[0]), 2)

    fk, wk, al = _arr("free_knee"), _arr("wt_knee"), _arr("ankle_lift")
    pit, com, tms = _arr("pitch_range"), _arr("com_drop"), _arr("t")

    acc = np.asarray(accent_times) if accent_times is not None else np.array([])
    if len(acc) and beat_dur > 0:
        aligned = np.array([float(np.min(np.abs(acc - t))) <= 0.5 * beat_dur for t in tms])
    else:
        aligned = np.zeros(len(tms), dtype=bool)

    def _split_mean(a):
        al_v = a[aligned & ~np.isnan(a)]
        no_v = a[(~aligned) & ~np.isnan(a)]
        return (round(float(np.mean(al_v)), 1) if len(al_v) else 0.0,
                round(float(np.mean(no_v)), 1) if len(no_v) else 0.0)

    wt_acc, wt_non = _split_mean(wk)
    fk_acc, fk_non = _split_mean(fk)

    return {
        "step_count":             len(recs),
        "dist_free_knee":         _dist(fk),
        "dist_weighted_knee":     _dist(wk),
        "dist_ankle_lift":        _dist(al),
        "accent_n":               int(aligned.sum()),
        "wt_knee_accent":         wt_acc,
        "wt_knee_nonaccent":      wt_non,
        "free_knee_accent":       fk_acc,
        "free_knee_nonaccent":    fk_non,
        "coupling_lift_flex":     _corr(al, fk),    # foot-lift vs free-leg prep flex
        "coupling_pitch_flex":    _corr(pit, fk),   # body pitch vs free-leg prep flex
        "coupling_sink_comdrop":  _corr(wk, com),   # standing-leg sink vs COM drop
    }


# ---------------------------------------------------------------------------
# 1. Leg Action
# ---------------------------------------------------------------------------

def compute_leg_action(frames: list, dancer_id: int, fps: float) -> dict:
    """
    Metrics:
        step_count_total        int
        step_count_articulated  int   — heel lifts clear of ground
        step_count_weight_only  int   — in-place shift, no heel lift (triple & counts, anchors)
        articulated_pct         float — % of steps that are articulated
        steps_per_minute        float — all weight changes per minute
        knee_flex_mean          float — normalised knee-hip distance
        knee_flex_max           float
        knee_flex_at_articulated float — knee flex measured near articulated steps only
        rise_fall_typical       float — median bounce amplitude (steady step-to-step rise/fall)
        rise_fall_dynamic       float — 95th-percentile bounce (occasional dramatic level changes)
        rise_fall_rhythm_hz     float
        triple_step_count       int   — articulated→weight_only→articulated in ~0.6 s
        one_foot_pct            float — % of frames in 1-foot balance (one ankle elevated)
        two_foot_pct            float — % of frames in 2-foot balance (both feet down)
    """
    duration = frames[-1]["time_sec"] - frames[0]["time_sec"] if frames else 1.0

    step_data  = detect_step_events(frames, dancer_id, fps)
    all_times  = step_data["all"]
    art_times  = step_data["articulated"]
    wo_times   = step_data["weight_only"]

    step_count   = len(all_times)
    spm          = (step_count / duration) * 60 if duration > 0 else 0.0
    art_pct      = round(100.0 * len(art_times) / max(step_count, 1), 1)

    # Knee flexion — all frames
    bh_samples, flex_all, flex_art = [], [], []
    art_set = set(round(t, 4) for t in art_times)

    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        bh  = body_height(kps)
        if bh < 10:
            continue
        bh_samples.append(bh)
        for knee_i, hip_i in [(KP["left_knee"], KP["left_hip"]),
                               (KP["right_knee"], KP["right_hip"])]:
            if kps[knee_i, 2] >= CONF_MIN and kps[hip_i, 2] >= CONF_MIN:
                fl = abs(kps[knee_i, 1] - kps[hip_i, 1]) / bh
                flex_all.append(fl)
                # Attribute to articulated steps if this frame is near one
                t = f["time_sec"]
                if any(abs(t - at) <= 0.15 for at in art_times):
                    flex_art.append(fl)

    knee_flex_mean        = round(float(np.mean(flex_all)), 3)  if flex_all else 0.0
    knee_flex_max         = round(float(np.max(flex_all)),  3)  if flex_all else 0.0
    knee_flex_articulated = round(float(np.mean(flex_art)), 3)  if flex_art else 0.0

    bh_mean_all = float(np.mean(bh_samples)) if bh_samples else 1.0
    art_quality = _articulation_quality(frames, dancer_id, fps, step_data, bh_mean_all)

    # Rise/fall vertical oscillation
    # Two complementary metrics:
    #   rise_fall_typical   = median Hilbert envelope amplitude — bounce on average steps
    #   rise_fall_dynamic   = 95th-percentile envelope        — biggest level changes
    # The std-based aggregate is dominated by a few extreme moments (drops, hits,
    # sit-downs) and so masks how much steady-state bounce is in the dance. The
    # split lets us tell "do you bounce more on every step?" apart from "do you
    # commit to dramatic level changes for hits?".
    t_c, _, y_c = _center_series(frames, dancer_id)
    rise_fall_typical = rise_fall_dynamic = rise_fall_rhythm_hz = 0.0
    if len(y_c) > SMOOTH_WINDOW + 2:
        from scipy.signal import hilbert as _hilbert
        bh_mean = float(np.mean(bh_samples)) if bh_samples else 1.0
        y_sm    = _smooth(y_c)
        dt      = float(np.mean(np.diff(t_c))) if len(t_c) > 1 else 1.0 / fps
        fs_rf   = 1.0 / dt
        y_bp    = _bandpass(y_sm, fs_rf, lo=0.5, hi=4.0)
        envelope = np.abs(_hilbert(y_bp)) / bh_mean
        rise_fall_typical = round(float(np.median(envelope)), 4)
        rise_fall_dynamic = round(float(np.percentile(envelope, 95)), 4)
        fft_mag = np.abs(np.fft.rfft(y_bp))
        fft_frq = np.fft.rfftfreq(len(y_bp), d=dt)
        # find peak in the 0.3–4 Hz dance band
        mask_rf = (fft_frq >= 0.3) & (fft_frq <= 4.0)
        if mask_rf.any():
            rise_fall_rhythm_hz = round(float(fft_frq[mask_rf][np.argmax(fft_mag[mask_rf])]), 3)
        else:
            rise_fall_rhythm_hz = round(float(fft_frq[np.argmax(fft_mag[1:]) + 1]), 3)

    # Triple step: articulated → weight_only → articulated within ~0.6 s
    # The middle event must be weight_only; outer two must be articulated.
    TRIPLE_WINDOW = 0.65
    triple_count  = 0
    wo_set        = set(wo_times)
    for i in range(len(all_times) - 2):
        t0, t1, t2 = all_times[i], all_times[i + 1], all_times[i + 2]
        if (t2 - t0) <= TRIPLE_WINDOW and t1 in wo_set:
            triple_count += 1

    return {
        "step_count_total":                   step_count,
        "step_count_articulated":             len(art_times),
        "step_count_weight_only":             len(wo_times),
        "step_count_articulated_traveling":   len(step_data.get("articulated_traveling", [])),
        "step_count_articulated_in_place":    len(step_data.get("articulated_in_place", [])),
        "step_count_weight_only_traveling":   len(step_data.get("weight_only_traveling", [])),
        "step_count_weight_only_in_place":    len(step_data.get("weight_only_in_place", [])),
        "articulated_pct":          art_pct,
        "steps_per_minute":         round(spm, 1),
        "knee_flex_mean":           knee_flex_mean,
        "knee_flex_max":            knee_flex_max,
        "knee_flex_at_articulated": knee_flex_articulated,
        "rise_fall_typical":        rise_fall_typical,
        "rise_fall_dynamic":        rise_fall_dynamic,
        "rise_fall_rhythm_hz":      rise_fall_rhythm_hz,
        "triple_step_count":        triple_count,
        "one_foot_pct":             step_data.get("one_foot_pct", 0.0),
        "two_foot_pct":             step_data.get("two_foot_pct", 0.0),
        "one_foot_airborne_pct":    step_data.get("one_foot_airborne_pct", 0.0),
        "ball_foot_pct":            step_data.get("ball_foot_pct", 0.0),
        "foot_kps_used":            step_data.get("foot_kps_used", False),
        **art_quality,              # art_* articulation-quality metrics
        "step_data":                step_data,   # kept for musicality
    }


# ---------------------------------------------------------------------------
# 2. Body Action
# ---------------------------------------------------------------------------

def compute_body_action(frames: list, dancer_id: int, fps: float) -> dict:
    """
    Three aspects of body action:

    PITCH  — forward/backward lean along the slot (torso tilt in image plane).
             For a side-on camera the torso's left/right angle from vertical is the
             pitch angle — positive = pitched forward in slot, negative = back.
        pitch_range_deg     float  range of pitch angles seen (degrees)
        pitch_rhythm_hz     float  dominant frequency of pitch oscillation

    FLUIDITY — sequential propagation of lateral motion through the kinematic chain.
             Cross-correlation lag between the hip, shoulder, and head lateral (x)
             positions reveals whether each segment follows the one below it with a
             delay (bottom-up wave = positive lag) or all move together (block body).
        hip_shoulder_lag_ms   float  ms shoulder lateral motion lags hip (+ = sequential)
        shoulder_head_lag_ms  float  ms head lateral motion lags shoulder
        motion_smoothness     float  0–1; fraction of hip velocity energy below 2 Hz
                                     (higher = smoother, more continuous movement)

    SWAY — side-to-side body tilt: shoulder and hip line angles from horizontal.
        shoulder_tilt_range_deg  float  range of shoulder line tilt (degrees)
        hip_tilt_range_deg       float  range of hip line tilt (degrees)
        sway_rhythm_hz           float  dominant frequency of shoulder tilt changes
        upper_lower_sway_dissoc  float  mean |shoulder_tilt − hip_tilt| (degrees);
                                        higher = upper body tilts independently of hips
    """
    bhs = []
    pitch_angles: list[float] = []
    pitch_times:  list[float] = []
    sh_angles:    list[float] = []
    hi_angles:    list[float] = []
    sway_times:   list[float] = []

    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        bh  = body_height(kps)
        if bh < 10:
            continue
        bhs.append(bh)
        t = f["time_sec"]

        # Pitch: torso lean angle in image plane (head→hip vector from vertical)
        if (kps[KP["head"], 2] >= CONF_MIN
                and kps[KP["left_hip"], 2] >= CONF_MIN
                and kps[KP["right_hip"], 2] >= CONF_MIN):
            pitch_angles.append(torso_angle(kps))
            pitch_times.append(t)

        # Sway: shoulder/hip line tilt — only when both lines are wide enough
        # to be reliable (foreshortening filter)
        ls, rs = kps[KP["left_shoulder"]], kps[KP["right_shoulder"]]
        lh, rh = kps[KP["left_hip"]],      kps[KP["right_hip"]]
        sh_ok = (ls[2] >= CONF_MIN and rs[2] >= CONF_MIN
                 and abs(rs[0] - ls[0]) > bh * 0.10)
        hi_ok = (lh[2] >= CONF_MIN and rh[2] >= CONF_MIN
                 and abs(rh[0] - lh[0]) > bh * 0.08)
        if sh_ok and hi_ok:
            sh_angles.append(shoulder_angle(kps))
            hi_angles.append(hip_angle(kps))
            sway_times.append(t)

    bh_mean = float(np.mean(bhs)) if bhs else 1.0

    # ------------------------------------------------- 3D UPGRADE (if lifted)
    # The 2D projections above conflate pitch with lateral lean whenever the
    # camera isn't exactly side-on, and are blind to axial rotation (a rotating
    # shoulder line foreshortens in the image instead of tilting — those frames
    # are even filtered out above). With lifted 3D joints:
    #   * clip-level "up" = the average pelvis→thorax direction (a dancer is
    #     upright on average), so no camera-tilt assumption is needed;
    #   * per frame, the hip line (anatomical right−left, so the frame stays
    #     body-fixed through turns) gives the lateral axis; up × lateral gives
    #     body-forward;
    #   * pitch  = torso angle in the body's sagittal plane (true fwd/back),
    #     tilt   = shoulder/hip line elevation out of the horizontal plane,
    #     rotation = signed angle between the shoulder and hip lines projected
    #     onto the horizontal plane — upper/lower AXIAL rotation, which 2D
    #     fundamentally cannot see.
    # The pitch/sway series computed above are then replaced so the shared
    # aggregation below runs on the truer signals.
    body_angle_source = "2d"
    rot_angles: list[float] = []
    if _pose3d_available(frames, dancer_id):
        ts3, torsos, sh_lines, hip_lines = [], [], [], []
        for f in frames:
            k3 = _kps3d(f, dancer_id)
            if k3 is None:
                continue
            ts3.append(f["time_sec"])
            torsos.append(k3[KP3D["thorax"]] - k3[KP3D["pelvis"]])
            sh_lines.append(k3[KP3D["right_shoulder"]] - k3[KP3D["left_shoulder"]])
            hip_lines.append(k3[KP3D["right_hip"]] - k3[KP3D["left_hip"]])
        if len(ts3) > SMOOTH_WINDOW + 2:
            torsos    = np.asarray(torsos)
            sh_lines  = np.asarray(sh_lines)
            hip_lines = np.asarray(hip_lines)
            up = torsos.mean(axis=0)
            up /= max(float(np.linalg.norm(up)), 1e-9)

            def _unit(a):
                return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)

            hip_h = hip_lines - np.outer(hip_lines @ up, up)   # horizontal components
            sh_h  = sh_lines  - np.outer(sh_lines  @ up, up)
            ok = ((np.linalg.norm(hip_h, axis=1) > 1e-6)
                  & (np.linalg.norm(sh_h, axis=1) > 1e-6)
                  & (np.linalg.norm(torsos, axis=1) > 1e-6))

            lat = _unit(hip_h)                 # body lateral axis
            fwd = np.cross(up, lat)            # body forward axis (right-handed)
            pitch3 = np.degrees(np.arctan2(np.einsum("ij,ij->i", torsos, fwd),
                                           torsos @ up))
            sh_tilt3  = np.degrees(np.arcsin(np.clip(_unit(sh_lines)  @ up, -1, 1)))
            hip_tilt3 = np.degrees(np.arcsin(np.clip(_unit(hip_lines) @ up, -1, 1)))
            sh_hn, hip_hn = _unit(sh_h), _unit(hip_h)
            rot3 = np.degrees(np.arctan2(np.cross(hip_hn, sh_hn) @ up,
                                         np.einsum("ij,ij->i", hip_hn, sh_hn)))

            ts3 = np.asarray(ts3)
            pitch_times,  pitch_angles = list(ts3[ok]), list(pitch3[ok])
            sway_times = list(ts3[ok])
            sh_angles, hi_angles = list(sh_tilt3[ok]), list(hip_tilt3[ok])
            rot_angles = list(rot3[ok])
            body_angle_source = "3d"

    # ------------------------------------------------------------------ PITCH
    pitch_range_deg = 0.0
    pitch_rhythm_hz = 0.0
    if len(pitch_angles) > SMOOTH_WINDOW + 2:
        t_p   = np.array(pitch_times)
        p_sm  = _smooth(np.array(pitch_angles))
        pitch_range_deg = float(_circular_range(p_sm))
        dt    = float(np.mean(np.diff(t_p))) if len(t_p) > 1 else 1.0 / fps
        p_bp  = _bandpass(p_sm, 1.0 / dt, lo=0.1, hi=4.0)
        fft_m = np.abs(np.fft.rfft(p_bp))
        fft_f = np.fft.rfftfreq(len(p_bp), d=dt)
        mask_p = (fft_f >= 0.1) & (fft_f <= 4.0)
        pitch_rhythm_hz = float(fft_f[mask_p][np.argmax(fft_m[mask_p])]) if mask_p.any() else float(fft_f[np.argmax(fft_m[1:]) + 1])

    # --------------------------------------------------------------- FLUIDITY
    # Use vertical (y) position: the WCS bounce/pulse propagates upward through
    # the body (hips → shoulders → head), making vertical lag detectable side-on.
    # Image y increases downward, so invert so that "up" is positive.
    def _vert_y(kp_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ts, ys = [], []
        for f in frames:
            if dancer_id not in f["dancers"]:
                continue
            kps = f["dancers"][dancer_id]
            vals = [-kps[i, 1] for i in kp_indices if kps[i, 2] >= CONF_MIN]
            if vals:
                ts.append(f["time_sec"])
                ys.append(float(np.mean(vals)))
        return np.array(ts), np.array(ys)

    t_hip, x_hip = _vert_y([KP["left_hip"],      KP["right_hip"]])
    t_sh,  x_sh  = _vert_y([KP["left_shoulder"], KP["right_shoulder"]])
    t_hd,  x_hd  = _vert_y([KP["head"]])

    def _xcorr_lag_ms(t_a, x_a, t_b, x_b, max_lag_s: float = 0.35) -> float:
        """
        Lag (ms) by which signal b follows signal a, estimated via cross-correlation
        on a common time grid.  Positive = b lags a.
        """
        t0  = max(t_a[0],  t_b[0])
        t1  = min(t_a[-1], t_b[-1])
        if t1 - t0 < 1.0:
            return 0.0
        # Upsample 4× for sub-frame lag resolution (~10 ms at 25 fps)
        n       = int((t1 - t0) * fps * 4) + 1
        t_com   = np.linspace(t0, t1, n)
        dt_com  = float(np.mean(np.diff(t_com)))
        fs_loc  = 1.0 / dt_com if dt_com > 0 else fps * 4
        sig_a   = _bandpass(np.interp(t_com, t_a, x_a), fs_loc)
        sig_b   = _bandpass(np.interp(t_com, t_b, x_b), fs_loc)
        sig_a  -= np.mean(sig_a)
        sig_b  -= np.mean(sig_b)
        corr    = _correlate(sig_b, sig_a, mode="full", method="fft")
        lags    = (np.arange(len(corr)) - (len(sig_a) - 1)) * dt_com
        mask    = np.abs(lags) <= max_lag_s
        best    = lags[mask][np.argmax(corr[mask])]
        return round(float(best * 1000), 1)

    hip_shoulder_lag_ms  = 0.0
    shoulder_head_lag_ms = 0.0
    if len(t_hip) > 20 and len(t_sh) > 20:
        hip_shoulder_lag_ms  = _xcorr_lag_ms(t_hip, x_hip, t_sh, x_sh)
    if len(t_sh) > 20 and len(t_hd) > 20:
        shoulder_head_lag_ms = _xcorr_lag_ms(t_sh, x_sh, t_hd, x_hd)

    # Smoothness: fraction of hip lateral position energy below 3 Hz (detrended)
    motion_smoothness = 0.0
    if len(t_hip) > SMOOTH_WINDOW + 2:
        dt_arr  = np.diff(t_hip)
        dt_mean = float(np.mean(dt_arr))
        x_dt    = _detrend_fft(_smooth(x_hip))
        fft_m   = np.abs(np.fft.rfft(x_dt))
        fft_f   = np.fft.rfftfreq(len(x_dt), d=dt_mean)
        total   = float(np.sum(fft_m[1:] ** 2))
        low     = float(np.sum(fft_m[(fft_f > 0) & (fft_f < 3.0)] ** 2))
        motion_smoothness = round(low / (total + 1e-9), 3)

    # ------------------------------------------------------------------- SWAY
    sh_arr = np.array(sh_angles)
    hi_arr = np.array(hi_angles)

    shoulder_tilt_range = _circular_range(sh_arr) if len(sh_arr) else 0.0
    hip_tilt_range      = _circular_range(hi_arr) if len(hi_arr) else 0.0

    sway_rhythm_hz = 0.0
    if len(sway_times) > SMOOTH_WINDOW + 2:
        t_sw  = np.array(sway_times)
        sh_sm = _smooth(sh_arr)
        dt    = float(np.mean(np.diff(t_sw))) if len(t_sw) > 1 else 1.0 / fps
        sh_bp  = _bandpass(sh_sm, 1.0 / dt, lo=0.1, hi=4.0)
        fft_m  = np.abs(np.fft.rfft(sh_bp))
        fft_f  = np.fft.rfftfreq(len(sh_sm), d=dt)
        mask_s = (fft_f >= 0.1) & (fft_f <= 4.0)
        sway_rhythm_hz = float(fft_f[mask_s][np.argmax(fft_m[mask_s])]) if mask_s.any() else float(fft_f[np.argmax(fft_m[1:]) + 1])

    upper_lower_sway_dissoc = 0.0
    if len(sh_arr):
        raw  = sh_arr - hi_arr
        diss = np.abs((raw + 180) % 360 - 180)
        upper_lower_sway_dissoc = float(np.mean(diss))

    # -------------------------------------------------- AXIAL ROTATION (3D only)
    upper_lower_rotation_mean = 0.0
    upper_lower_rotation_p90  = 0.0
    if len(rot_angles) > SMOOTH_WINDOW + 2:
        rot_abs = np.abs(_smooth(np.array(rot_angles)))
        upper_lower_rotation_mean = float(np.mean(rot_abs))
        upper_lower_rotation_p90  = float(np.percentile(rot_abs, 90))

    # 3D ranges: robust percentile spread instead of min-max. The min-max
    # (circular) range is defined by a handful of outlier frames — a single
    # glitchy spin moment sets the number for the whole clip (2D tilt "ranges"
    # read 200°+ because of this). The 3D angles live in [-90, 90] with no wrap,
    # so a plain p97.5−p2.5 spread is meaningful and stable.
    if body_angle_source == "3d":
        def _robust_range(vals):
            a = _smooth(np.array(vals)) if len(vals) >= SMOOTH_WINDOW else np.array(vals)
            return float(np.percentile(a, 97.5) - np.percentile(a, 2.5)) if len(a) else 0.0
        if pitch_angles:
            pitch_range_deg = _robust_range(pitch_angles)
        shoulder_tilt_range = _robust_range(sh_angles)
        hip_tilt_range      = _robust_range(hi_angles)

    return {
        "body_angle_source":              body_angle_source,
        "upper_lower_rotation_mean_deg":  round(upper_lower_rotation_mean, 1),
        "upper_lower_rotation_p90_deg":   round(upper_lower_rotation_p90, 1),
        "pitch_range_deg":           round(pitch_range_deg, 1),
        "pitch_rhythm_hz":           round(pitch_rhythm_hz, 3),
        "hip_shoulder_lag_ms":       hip_shoulder_lag_ms,
        "shoulder_head_lag_ms":      shoulder_head_lag_ms,
        "motion_smoothness":         motion_smoothness,
        "shoulder_tilt_range_deg":   round(shoulder_tilt_range, 1),
        "hip_tilt_range_deg":        round(hip_tilt_range, 1),
        "sway_rhythm_hz":            round(sway_rhythm_hz, 3),
        "upper_lower_sway_dissoc":   round(upper_lower_sway_dissoc, 1),
    }


# ---------------------------------------------------------------------------
# 3. Weight & Countering (partnership)
# ---------------------------------------------------------------------------

def _connection_type(kps_a: np.ndarray, kps_b: np.ndarray, bh_a: float, bh_b: float) -> tuple[str, np.ndarray]:
    """
    Detect the connection point between two dancers.

    Strategy:
      1. Check wrist–wrist distance (hand-to-hand connection)
      2. Check each dancer's wrist against the other's torso centre (hand-to-back/front)
      3. Fall back to torso-midpoint distance

    Returns:
        (connection_type_str, connection_point_xy)
    """
    scale = (bh_a + bh_b) / 2.0

    def pt(kps, idx):
        return kps[idx, :2]

    def torso_centre(kps):
        pts = [kps[i, :2] for i in [KP["left_shoulder"], KP["right_shoulder"],
                                     KP["left_hip"],      KP["right_hip"]]
               if kps[i, 2] >= CONF_MIN]
        return np.mean(pts, axis=0) if pts else kps[KP["left_hip"], :2]

    wrists_a = [kps_a[KP["left_wrist"]], kps_a[KP["right_wrist"]]]
    wrists_b = [kps_b[KP["left_wrist"]], kps_b[KP["right_wrist"]]]

    # Hand-to-hand: closest wrist pair
    best_hh_dist = np.inf
    best_hh_pt   = None
    for wa in wrists_a:
        for wb in wrists_b:
            if wa[2] < CONF_MIN or wb[2] < CONF_MIN:
                continue
            d = np.linalg.norm(wa[:2] - wb[:2]) / scale
            if d < best_hh_dist:
                best_hh_dist = d
                best_hh_pt   = (wa[:2] + wb[:2]) / 2.0

    if best_hh_dist <= HAND_HAND_FRAC:
        return "hand-hand", best_hh_pt

    # Hand-to-body: wrist of A near torso of B (or vice-versa)
    tc_a = torso_centre(kps_a)
    tc_b = torso_centre(kps_b)

    best_hb_dist = np.inf
    best_hb_pt   = None
    for wa in wrists_a:
        if wa[2] < CONF_MIN:
            continue
        d = np.linalg.norm(wa[:2] - tc_b) / scale
        if d < best_hb_dist:
            best_hb_dist = d
            best_hb_pt   = (wa[:2] + tc_b) / 2.0
    for wb in wrists_b:
        if wb[2] < CONF_MIN:
            continue
        d = np.linalg.norm(wb[:2] - tc_a) / scale
        if d < best_hb_dist:
            best_hb_dist = d
            best_hb_pt   = (wb[:2] + tc_a) / 2.0

    if best_hb_dist <= HAND_BACK_FRAC:
        return "hand-body", best_hb_pt

    # Fallback: torso-to-torso midpoint
    return "none", (tc_a + tc_b) / 2.0


def _slot_axis(frames: list, dancer_ids: list[int]) -> tuple[np.ndarray, float]:
    """
    Estimate the slot axis: the dominant spatial direction the partnership occupies.

    PCA over the stacked per-frame hip centres of BOTH dancers across the clip.
    For a roughly side-on camera this is the line the dancers travel along, which
    is what 'horizontal'/'down the slot' means for posts and travel decomposition.

    Returns (unit_vector_xy, angle_deg_from_horizontal). Falls back to the image
    horizontal axis (1, 0) when there is insufficient data.
    """
    if len(dancer_ids) < 2:
        return np.array([1.0, 0.0]), 0.0
    id_a, id_b = dancer_ids[0], dancer_ids[1]
    pts: list[list[float]] = []
    for f in frames:
        d = f.get("dancers", {})
        for did in (id_a, id_b):
            if did in d:
                c = get_center(d[did])
                if c[2] >= CONF_MIN:
                    pts.append([float(c[0]), float(c[1])])
    if len(pts) < 10:
        return np.array([1.0, 0.0]), 0.0
    arr = np.array(pts)
    arr = arr - arr.mean(axis=0)
    cov = np.cov(arr.T)
    evals, evecs = np.linalg.eig(cov)
    principal = np.real(evecs[:, int(np.argmax(np.real(evals)))])
    norm = np.linalg.norm(principal)
    if norm < 1e-9:
        return np.array([1.0, 0.0]), 0.0
    u = principal / norm
    angle = float(np.degrees(np.arctan2(u[1], u[0])))
    return u, angle


def _detect_posts(
    times: np.ndarray,
    conn_xy: np.ndarray,   # (N, 2) connection point positions in pixels
    center_a: np.ndarray,  # (N, 2) dancer A hip centres
    center_b: np.ndarray,  # (N, 2) dancer B hip centres
    bh_mean: float,
    slot_axis: np.ndarray,
) -> dict:
    """
    Detect 'post' moments: periods when the connection point stops traveling
    ALONG THE SLOT, giving both dancers a fixed anchor to stretch away from or
    compress into.  Stretch and compression legitimately move the hand vertically
    (and slightly perpendicular), so stillness is measured only along the slot
    axis — vertical/perpendicular hand motion no longer breaks a post.

    A post is detected when the smoothed connection-point slot-axis speed drops
    below POST_SPEED_THRESH (in body-heights per second) for at least
    MIN_POST_DURATION seconds.  For each detected post, the partner distance at
    initiation is used as the baseline, and the maximum stretch / compression
    achieved in the following POST_MEASURE_WIN seconds is recorded.

    Returns:
        {
            "post_count":                int,
            "post_max_stretch_mean":     float,  # mean of per-post max stretch (BH)
            "post_max_compression_mean": float,  # mean of per-post max compression (BH)
        }
    """
    POST_SPEED_THRESH = 0.20   # BH/s — below this the connection point is "posted"
    MIN_POST_DURATION = 0.18   # seconds of stillness required to count as a post
    POST_MEASURE_WIN  = 1.5    # seconds after post start to measure stretch/compression

    if len(times) < 10:
        return {"post_count": 0, "post_stretch_leading": 0,
                "post_compression_leading": 0, "post_max_stretch_mean": 0.0,
                "post_max_compression_mean": 0.0}

    # Smooth connection-point path then compute speed ALONG THE SLOT only.
    # Project the per-frame displacement onto the slot unit vector so that
    # vertical/perpendicular motion (from stretch/compression) is ignored.
    conn_x_sm = _smooth(conn_xy[:, 0]) if len(conn_xy) >= SMOOTH_WINDOW else conn_xy[:, 0]
    conn_y_sm = _smooth(conn_xy[:, 1]) if len(conn_xy) >= SMOOTH_WINDOW else conn_xy[:, 1]

    u  = np.asarray(slot_axis, dtype=float)
    nu = np.linalg.norm(u)
    u  = u / nu if nu > 1e-9 else np.array([1.0, 0.0])

    dt_arr = np.diff(times)
    d_slot = np.diff(conn_x_sm) * u[0] + np.diff(conn_y_sm) * u[1]
    speed  = np.abs(d_slot) / (dt_arr * bh_mean + 1e-6)
    speed_sm = _smooth(speed) if len(speed) >= SMOOTH_WINDOW else speed
    t_mid    = (times[:-1] + times[1:]) / 2.0  # timestamps for each speed sample

    # Scan for stationary periods
    raw_posts: list[tuple[int, int]] = []
    in_post = False
    post_start_i = 0
    for i, sp in enumerate(speed_sm):
        if not in_post and sp < POST_SPEED_THRESH:
            in_post = True
            post_start_i = i
        elif in_post and sp >= POST_SPEED_THRESH:
            if t_mid[i - 1] - t_mid[post_start_i] >= MIN_POST_DURATION:
                raw_posts.append((post_start_i, i - 1))
            in_post = False
    if in_post and t_mid[-1] - t_mid[post_start_i] >= MIN_POST_DURATION:
        raw_posts.append((post_start_i, len(speed_sm) - 1))

    if not raw_posts:
        return {"post_count": 0, "post_stretch_leading": 0,
                "post_compression_leading": 0, "post_max_stretch_mean": 0.0,
                "post_max_compression_mean": 0.0}

    # For each post: measure stretch / compression relative to partner distance at
    # initiation, and classify the post by which dominates afterwards (does the
    # dancer anchor to SEND/stretch, or to RECEIVE/compress?).
    stretches:     list[float] = []
    compressions:  list[float] = []
    stretch_leading     = 0
    compression_leading = 0

    for start_i, _end_i in raw_posts:
        post_t = float(t_mid[start_i])

        # Index into the original (non-speed) arrays closest to post start
        orig_i = int(np.searchsorted(times, post_t))
        if orig_i >= len(times):
            continue

        d0 = float(np.linalg.norm(center_a[orig_i] - center_b[orig_i])) / bh_mean

        win = (times >= post_t) & (times <= post_t + POST_MEASURE_WIN)
        if win.sum() < 2:
            continue

        dists = np.linalg.norm(center_a[win] - center_b[win], axis=1) / bh_mean
        s = max(0.0, float(np.max(dists)) - d0)
        c = max(0.0, d0 - float(np.min(dists)))
        stretches.append(s)
        compressions.append(c)
        if s >= c:
            stretch_leading += 1
        else:
            compression_leading += 1

    if not stretches:
        return {"post_count": 0, "post_stretch_leading": 0,
                "post_compression_leading": 0, "post_max_stretch_mean": 0.0,
                "post_max_compression_mean": 0.0}

    return {
        "post_count":                len(stretches),
        "post_stretch_leading":      stretch_leading,
        "post_compression_leading":  compression_leading,
        "post_max_stretch_mean":     round(float(np.mean(stretches)),     3),
        "post_max_compression_mean": round(float(np.mean(compressions)),  3),
    }


def compute_weight_countering(
    frames: list,
    dancer_ids: list[int],
    fps: float,
    slot_axis: np.ndarray | None = None,
) -> dict:
    """
    Partnership metrics relative to the detected connection point.

    Metrics:
        connection_type_counts       dict   (% of frames for each type)
        partner_distance_mean        float  (mean centre-centre distance, normalised)
        partner_distance_std         float
        stretch_pct                  float  (% time partners are extending away from connection)
        compression_pct              float  (% time partners are moving toward connection)
        lean_toward_conn_a           float  (mean lean angle of dancer A toward connection point)
        lean_toward_conn_b           float  (mean lean angle of dancer B toward connection point)
        counter_balance_pct          float  (% frames both dancers lean away from each other)
        slot_direction_deg           float  (dominant movement axis, degrees from horizontal)
        post_count                   int    (number of post moments detected)
        post_stretch_leading         int    (posts followed mainly by stretch — anchor to SEND)
        post_compression_leading     int    (posts followed mainly by compression — anchor to RECEIVE)
        post_max_stretch_mean        float  (mean peak stretch after a post, in body heights)
        post_max_compression_mean    float  (mean peak compression after a post, in body heights)
    """
    if len(dancer_ids) < 2:
        return {"error": "need 2 dancers for partnership metrics"}

    if slot_axis is None:
        slot_axis, _ = _slot_axis(frames, dancer_ids)

    id_a, id_b = dancer_ids[0], dancer_ids[1]

    conn_types   = []
    distances    = []
    velocities   = []   # positive = stretching, negative = compressing
    lean_a_vals  = []
    lean_b_vals  = []
    counter_frames = 0
    total_frames   = 0
    bh_samples = []

    prev_dist = None
    slot_dx, slot_dy = [], []

    # Collected per-frame for post detection
    post_t_list:    list[float]     = []
    post_conn_list: list[np.ndarray] = []
    post_ca_list:   list[np.ndarray] = []
    post_cb_list:   list[np.ndarray] = []

    for f in frames:
        if id_a not in f["dancers"] or id_b not in f["dancers"]:
            continue
        kps_a = f["dancers"][id_a]
        kps_b = f["dancers"][id_b]
        bh_a  = body_height(kps_a)
        bh_b  = body_height(kps_b)
        scale = (bh_a + bh_b) / 2.0
        if scale < 10:
            continue
        bh_samples.append(scale)

        conn_type, conn_pt = _connection_type(kps_a, kps_b, bh_a, bh_b)
        conn_types.append(conn_type)

        c_a = get_center(kps_a)[:2]
        c_b = get_center(kps_b)[:2]
        dist = float(np.linalg.norm(c_a - c_b) / scale)
        distances.append(dist)

        if prev_dist is not None:
            velocities.append(dist - prev_dist)
        prev_dist = dist

        # Lean angle of each dancer toward the connection point
        # Positive = leaning toward conn_pt, negative = leaning away
        def lean_toward(kps, conn_pt):
            head = kps[KP["head"], :2]
            hip  = get_center(kps)[:2]
            torso_vec  = hip - head
            target_vec = conn_pt - head
            if np.linalg.norm(torso_vec) < 1 or np.linalg.norm(target_vec) < 1:
                return 0.0
            cos_a = np.dot(torso_vec, target_vec) / (
                np.linalg.norm(torso_vec) * np.linalg.norm(target_vec))
            return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))

        la = lean_toward(kps_a, conn_pt)
        lb = lean_toward(kps_b, conn_pt)
        lean_a_vals.append(la)
        lean_b_vals.append(lb)

        # Counter-balance: both dancers lean away from each other
        # Approximated by: torso angle of A points away from B's centre, and vice versa
        vec_ab = c_b - c_a
        vec_ba = -vec_ab
        ta     = get_center(kps_a)[:2] - kps_a[KP["head"], :2]
        tb     = get_center(kps_b)[:2] - kps_b[KP["head"], :2]
        # If torso vector (head→hips) is roughly opposite to direction toward partner,
        # the dancer is leaning away → counter-balance
        dot_a  = np.dot(ta, vec_ab)
        dot_b  = np.dot(tb, vec_ba)
        if dot_a > 0 and dot_b > 0:  # both hips toward each other = counter
            counter_frames += 1
        total_frames += 1

        # Slot direction: track centre-of-partnership movement
        slot_dx.append(float((c_a[0] + c_b[0]) / 2))
        slot_dy.append(float((c_a[1] + c_b[1]) / 2))

        # Collect for post detection — only when there is an active connection
        if conn_type != "none":
            post_t_list.append(f["time_sec"])
            post_conn_list.append(conn_pt.copy())
            post_ca_list.append(c_a.copy())
            post_cb_list.append(c_b.copy())

    # Connection type percentage
    from collections import Counter
    type_counts = Counter(conn_types)
    total_typed = max(len(conn_types), 1)
    conn_type_pcts = {k: round(100 * v / total_typed, 1) for k, v in type_counts.items()}

    velocities = np.array(velocities)
    stretch_pct     = float(100 * np.sum(velocities > 0.005) / max(len(velocities), 1))
    compression_pct = float(100 * np.sum(velocities < -0.005) / max(len(velocities), 1))

    # Slot direction: PCA on partnership centre trajectory
    slot_direction_deg = 0.0
    if len(slot_dx) > 10:
        pts = np.column_stack([slot_dx, slot_dy])
        pts -= pts.mean(axis=0)
        cov = np.cov(pts.T)
        evals, evecs = np.linalg.eig(cov)
        principal = evecs[:, np.argmax(evals)]
        slot_direction_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))

    # Post detection
    bh_mean = float(np.mean(bh_samples)) if bh_samples else 1.0
    post_metrics = {"post_count": 0, "post_stretch_leading": 0,
                    "post_compression_leading": 0, "post_max_stretch_mean": 0.0,
                    "post_max_compression_mean": 0.0}
    if len(post_t_list) >= 10:
        post_metrics = _detect_posts(
            np.array(post_t_list),
            np.array(post_conn_list),
            np.array(post_ca_list),
            np.array(post_cb_list),
            bh_mean,
            slot_axis,
        )

    return {
        "connection_type_pcts":      conn_type_pcts,
        "partner_distance_mean":     round(float(np.mean(distances)), 3) if distances else 0.0,
        "partner_distance_std":      round(float(np.std(distances)), 3)  if distances else 0.0,
        "stretch_pct":               round(stretch_pct, 1),
        "compression_pct":           round(compression_pct, 1),
        "lean_toward_conn_a":        round(float(np.mean(lean_a_vals)), 1) if lean_a_vals else 0.0,
        "lean_toward_conn_b":        round(float(np.mean(lean_b_vals)), 1) if lean_b_vals else 0.0,
        "counter_balance_pct":       round(100 * counter_frames / max(total_frames, 1), 1),
        "slot_direction_deg":        round(slot_direction_deg, 1),
        "post_count":                post_metrics["post_count"],
        "post_stretch_leading":      post_metrics["post_stretch_leading"],
        "post_compression_leading":  post_metrics["post_compression_leading"],
        "post_max_stretch_mean":     post_metrics["post_max_stretch_mean"],
        "post_max_compression_mean": post_metrics["post_max_compression_mean"],
    }


def compute_travel(
    frames: list,
    dancer_ids: list[int],
    fps: float,
    slot_axis: np.ndarray | None = None,
) -> dict:
    """
    Decompose partnership movement into three physically distinct travel types,
    all normalised to mean body height (BH):

      (a) couple_travel  — the couple relocating AROUND THE ROOM: path length and
                           range of the heavily-smoothed partnership centroid
                           (smoothing keeps sustained relocation, drops jitter).
      (b) slot_travel     — a dancer travelling DOWN THE SLOT: per-dancer range/path
                           of the dancer's centre projected onto the slot axis,
                           measured RELATIVE TO THE CENTROID so couple relocation
                           is not double-counted.
      (c) stretch/compression movement — how far centres move after a post — is the
                           existing post_max_stretch/compression (in weight_countering),
                           surfaced under the travel grouping by the report.

    Returns:
        {
          "slot_axis_deg":            float,
          "couple_travel_path_bh":    float,
          "couple_travel_range_bh":   float,
          "lead":   {"slot_travel_range_bh": float, "slot_travel_path_bh": float},
          "follow": {"slot_travel_range_bh": float, "slot_travel_path_bh": float},
        }
    Note: 'lead' is dancer_ids[0], 'follow' is dancer_ids[1] — the same role
    convention as leg_action_lead / leg_action_follow.
    """
    empty_side = {"slot_travel_range_bh": 0.0, "slot_travel_path_bh": 0.0}
    base = {"slot_axis_deg": 0.0, "couple_travel_path_bh": 0.0,
            "couple_travel_range_bh": 0.0,
            "lead": dict(empty_side), "follow": dict(empty_side)}
    if len(dancer_ids) < 2:
        return base

    id_a, id_b = dancer_ids[0], dancer_ids[1]
    if slot_axis is None:
        slot_axis, _ = _slot_axis(frames, dancer_ids)
    u  = np.asarray(slot_axis, dtype=float)
    nu = np.linalg.norm(u)
    u  = u / nu if nu > 1e-9 else np.array([1.0, 0.0])
    slot_angle = float(np.degrees(np.arctan2(u[1], u[0])))
    base["slot_axis_deg"] = round(slot_angle, 1)

    mid_x: list[float] = []
    mid_y: list[float] = []
    a_proj: list[float] = []   # lead  centre projected onto slot axis (room frame, absolute)
    b_proj: list[float] = []   # follow centre projected onto slot axis (room frame, absolute)
    bh_samples: list[float] = []
    for f in frames:
        d = f.get("dancers", {})
        if id_a not in d or id_b not in d:
            continue
        kps_a, kps_b = d[id_a], d[id_b]
        bh_a, bh_b = body_height(kps_a), body_height(kps_b)
        scale = (bh_a + bh_b) / 2.0
        if scale < 10:
            continue
        bh_samples.append(scale)
        c_a = get_center(kps_a)[:2]
        c_b = get_center(kps_b)[:2]
        mid = (c_a + c_b) / 2.0
        mid_x.append(float(mid[0]))
        mid_y.append(float(mid[1]))
        # Absolute slot-axis position of each dancer (room frame). Measuring this
        # absolutely — NOT relative to the 2-body centroid — keeps lead and follow
        # distinct (relative-to-centroid makes them exact mirror images).
        a_proj.append(float(np.dot(c_a, u)))
        b_proj.append(float(np.dot(c_b, u)))

    if len(mid_x) < SMOOTH_WINDOW + 2:
        return base

    bh_mean = float(np.mean(bh_samples)) if bh_samples else 1.0

    def _lowpass(arr: np.ndarray, win: int) -> np.ndarray:
        win = max(SMOOTH_WINDOW, win | 1)          # force odd, ≥ SMOOTH_WINDOW
        if len(arr) <= win:
            return _smooth(arr)
        return savgol_filter(arr, win, 2)

    # (a) couple around the room — STRONGLY low-passed centroid (≈1 s) so the metric
    # reflects sustained relocation around the floor, not per-step jitter. 'range'
    # (bounding extent) is the robust headline; 'path' (cumulative) is secondary.
    slow_win = int(fps) if fps and fps > 0 else SMOOTH_WINDOW
    mx = _lowpass(np.array(mid_x), slow_win)
    my = _lowpass(np.array(mid_y), slow_win)
    couple_path  = float(np.sum(np.sqrt(np.diff(mx) ** 2 + np.diff(my) ** 2))) / bh_mean
    couple_range = float(np.hypot(mx.max() - mx.min(), my.max() - my.min())) / bh_mean

    def _side(proj: list[float]) -> dict:
        p = _smooth(np.array(proj))
        rng  = float(p.max() - p.min()) / bh_mean
        path = float(np.sum(np.abs(np.diff(p)))) / bh_mean
        return {"slot_travel_range_bh": round(rng, 3),
                "slot_travel_path_bh":  round(path, 3)}

    return {
        "slot_axis_deg":          round(slot_angle, 1),
        "couple_travel_path_bh":  round(couple_path, 3),
        "couple_travel_range_bh": round(couple_range, 3),
        "lead":   _side(a_proj),
        "follow": _side(b_proj),
    }


# ---------------------------------------------------------------------------
# 4. Musicality / Timing
# ---------------------------------------------------------------------------

def _timing_stats(step_times: list[float], beat_times: np.ndarray,
                   half_beat: float) -> dict:
    """Compute on-beat %, timing consistency, and syncopation for a step list."""
    if not step_times:
        return {"on_beat_pct": 0.0, "consistency_ms": 0.0,
                "syncopation_pct": 0.0, "steps_per_beat": 0.0}
    offsets, on_beat, syncopated = [], 0, 0
    for st in step_times:
        nearest = beat_times[np.argmin(np.abs(beat_times - st))]
        offset  = st - nearest
        offsets.append(offset)
        if abs(offset) <= half_beat * 0.6:
            on_beat += 1
        if half_beat * 0.35 <= abs(offset) <= half_beat * 0.75:
            syncopated += 1
    n = len(step_times)
    return {
        "on_beat_pct":     round(100 * on_beat / n, 1),
        "consistency_ms":  round(float(np.std(offsets)) * 1000, 0),
        "syncopation_pct": round(100 * syncopated / n, 1),
        "steps_per_beat":  round(n / max(len(beat_times), 1), 2),
    }


def _arm_styling_metrics(frames: list, dancer_id: int, fps: float) -> dict:
    """
    Free arm styling for one dancer.

    shoulder_wrist_lag_ms  float  lateral shoulder→wrist cross-corr lag (ms).
                                  Positive = wrist follows shoulder = fluid wave.
                                  Near zero = stiff / block arm.
    body_arm_correlation   float  Pearson r between body-centre speed and wrist
                                  speed. Higher = arms amplify body movement.
    wrist_speed_mean       float  mean wrist speed, normalised to body height / s.
    """
    bh_vals = [body_height(f["dancers"][dancer_id])
               for f in frames if dancer_id in f["dancers"]
               and body_height(f["dancers"][dancer_id]) > 10]
    bh_mean = float(np.mean(bh_vals)) if bh_vals else 100.0

    def _lat(kp_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        ts, xs = [], []
        for f in frames:
            if dancer_id not in f["dancers"]:
                continue
            kps  = f["dancers"][dancer_id]
            vals = [kps[i, 0] for i in kp_indices if kps[i, 2] >= CONF_MIN]
            if vals:
                ts.append(f["time_sec"])
                xs.append(float(np.mean(vals)))
        return np.array(ts), np.array(xs)

    t_sh, x_sh = _lat([KP["left_shoulder"], KP["right_shoulder"]])

    wrist_data: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for kp in [KP["left_wrist"], KP["right_wrist"]]:
        t_w, x_w, y_w = _kp_series(frames, dancer_id, kp)
        wrist_data.append((t_w, x_w, y_w))

    # Shoulder → wrist lateral lag via FFT cross-correlation
    t_wr, x_wr, _ = max(wrist_data, key=lambda s: len(s[0]))
    sh_wr_lag_ms  = 0.0
    if len(t_sh) > 20 and len(t_wr) > 20:
        t0 = max(t_sh[0], t_wr[0]);  t1 = min(t_sh[-1], t_wr[-1])
        if t1 - t0 > 1.0:
            n     = int((t1 - t0) * fps) + 1
            t_c   = np.linspace(t0, t1, n)
            dt_c  = float(np.mean(np.diff(t_c)))
            fs_loc = 1.0 / dt_c if dt_c > 0 else fps
            s_sh  = _bandpass(np.interp(t_c, t_sh, x_sh), fs_loc)
            s_wr  = _bandpass(np.interp(t_c, t_wr, x_wr), fs_loc)
            s_sh -= np.mean(s_sh);  s_wr -= np.mean(s_wr)
            corr  = _correlate(s_wr, s_sh, mode="full", method="fft")
            lags  = (np.arange(len(corr)) - (len(s_sh) - 1)) * dt_c
            mask  = np.abs(lags) <= 0.5    # search ±500 ms
            best  = lags[mask][np.argmax(corr[mask])]
            sh_wr_lag_ms = round(float(best * 1000), 1)

    # Wrist speed (both wrists, normalised to BH/s)
    wrist_speeds: list[float] = []
    for t_w, x_w, y_w in wrist_data:
        if len(t_w) < 3:
            continue
        x_sm = _smooth(x_w) if len(x_w) >= SMOOTH_WINDOW else x_w
        y_sm = _smooth(y_w) if len(y_w) >= SMOOTH_WINDOW else y_w
        spd  = np.sqrt(np.diff(x_sm)**2 + np.diff(y_sm)**2) / (np.diff(t_w) * bh_mean + 1e-6)
        wrist_speeds.extend(spd.tolist())

    # Body–arm speed correlation: body-centre speed vs mean wrist speed
    body_arm_corr = 0.0
    t_c, x_c, y_c = _center_series(frames, dancer_id)
    if len(t_c) > SMOOTH_WINDOW + 2 and wrist_speeds:
        x_cs  = _smooth(x_c);  y_cs  = _smooth(y_c)
        bspd  = np.sqrt(np.diff(x_cs)**2 + np.diff(y_cs)**2) / (np.diff(t_c) + 1e-6)
        t_bp  = (t_c[:-1] + t_c[1:]) / 2.0
        arm_interps: list[np.ndarray] = []
        for t_w, x_w, y_w in wrist_data:
            if len(t_w) < 3:
                continue
            x_sm = _smooth(x_w) if len(x_w) >= SMOOTH_WINDOW else x_w
            y_sm = _smooth(y_w) if len(y_w) >= SMOOTH_WINDOW else y_w
            spd  = np.sqrt(np.diff(x_sm)**2 + np.diff(y_sm)**2) / (np.diff(t_w) + 1e-6)
            t_sp = (t_w[:-1] + t_w[1:]) / 2.0
            arm_interps.append(np.interp(t_bp, t_sp, spd))
        if arm_interps:
            arm_mean = np.mean(arm_interps, axis=0)
            n_min    = min(len(bspd), len(arm_mean))
            if n_min > 20:
                r, _ = _pearsonr(bspd[:n_min], arm_mean[:n_min])
                body_arm_corr = round(float(r), 3)

    return {
        "shoulder_wrist_lag_ms": sh_wr_lag_ms,
        "body_arm_correlation":  body_arm_corr,
        "wrist_speed_mean":      round(float(np.mean(wrist_speeds)) if wrist_speeds else 0.0, 3),
    }


def _channel_speed(frames: list, dancer_id: int, kp_indices: list[int],
                   bh_mean: float) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed 2-D speed (body-heights/s) of the mean position of a keypoint group.

    Returns (mid_times, speed) — empty arrays if too few confident samples.
    """
    ts, xs, ys = [], [], []
    for f in frames:
        if dancer_id not in f["dancers"]:
            continue
        kps = f["dancers"][dancer_id]
        vx = [kps[i, 0] for i in kp_indices if kps[i, 2] >= CONF_MIN]
        vy = [kps[i, 1] for i in kp_indices if kps[i, 2] >= CONF_MIN]
        if vx:
            ts.append(f["time_sec"])
            xs.append(float(np.mean(vx)))
            ys.append(float(np.mean(vy)))
    if len(ts) < 3:
        return np.array([]), np.array([])
    t = np.array(ts)
    x_sm = _smooth(np.array(xs)) if len(xs) >= SMOOTH_WINDOW else np.array(xs)
    y_sm = _smooth(np.array(ys)) if len(ys) >= SMOOTH_WINDOW else np.array(ys)
    spd  = np.sqrt(np.diff(x_sm)**2 + np.diff(y_sm)**2) / (np.diff(t) * bh_mean + 1e-6)
    t_mid = (t[:-1] + t[1:]) / 2.0
    return t_mid, spd


def _expression_channels(frames: list, dancer_id: int, fps: float) -> dict:
    """
    Per-channel movement-speed series used to detect musical expression through ANY
    channel — not just the free hand. WCS has no single 'correct' channel: a musical
    accent can be marked with a punctuated step, a chest pop, a free-arm accent, or a
    head accent (or by going still to frame the partner — see _accent_response).

    Channels (each a normalised 2-D speed series):
        feet  — ankles  (punctuated steps / footwork accents)
        chest — shoulders (chest pops / body accents)
        hands — wrists  (free-arm styling)
        head  — head    (head accents)

    Returns {channel: (mid_times, speed)}; channels with too few samples are omitted.
    """
    bh_vals = [body_height(f["dancers"][dancer_id])
               for f in frames if dancer_id in f["dancers"]
               and body_height(f["dancers"][dancer_id]) > 10]
    bh_mean = float(np.mean(bh_vals)) if bh_vals else 100.0

    groups = {
        "feet":  [KP["left_ankle"],    KP["right_ankle"]],
        "chest": [KP["left_shoulder"], KP["right_shoulder"]],
        "hands": [KP["left_wrist"],    KP["right_wrist"]],
        "head":  [KP["head"]],
    }
    out: dict = {}
    for name, idxs in groups.items():
        t, spd = _channel_speed(frames, dancer_id, idxs, bh_mean)
        if len(t) >= 3:
            out[name] = (t, spd)
    return out


def _burst_reference(t: np.ndarray, spd: np.ndarray, window: float) -> float:
    """
    A channel's *typical movement burst* at the given window scale: the median of the
    window-maximum speed sampled across the whole clip. Comparing an event's window
    peak to this answers "did the dancer move MORE here than in an ordinary moment?",
    which keeps the accent metric from saturating (every active moment beats the
    median speed, but only marked accents beat the typical burst).
    """
    if len(t) < 5:
        return float(np.median(spd)) + 1e-6 if len(spd) else 1e-6
    step    = max(window * 0.5, 0.2)
    centers = np.arange(float(t[0]), float(t[-1]), step)
    peaks   = []
    for c in centers:
        m = (t >= c - window) & (t <= c + window)
        if m.sum() >= 2:
            peaks.append(float(np.max(spd[m])))
    if len(peaks) >= 5:
        return float(np.median(peaks)) + 1e-6
    return float(np.median(spd)) + 1e-6


def _event_response(channels: dict, event_times: np.ndarray, window: float):
    """
    For each event time, the strongest channel response = max over channels of
    (peak speed in ±window) / (that channel's typical burst). 'Any channel counts.'

    Returns (ratios, chans): arrays length == len(event_times). ratios[i] is NaN when
    no channel had enough samples near event i; chans[i] is the dominant channel name
    (or None). A ratio ≈ 1 means an ordinary moment; ≥ ~1.5 means a marked accent.
    """
    refs = {name: _burst_reference(t, spd, window) for name, (t, spd) in channels.items()}
    ratios = np.full(len(event_times), np.nan)
    chans: list = [None] * len(event_times)
    for i, et in enumerate(event_times):
        best_ratio = 0.0
        best_ch = None
        for name, (t, spd) in channels.items():
            mask = (t >= et - window) & (t <= et + window)
            if mask.sum() < 2:
                continue
            ratio = float(np.max(spd[mask])) / refs[name]
            if ratio > best_ratio:
                best_ratio = ratio
                best_ch = name
        if best_ch is not None:
            ratios[i] = best_ratio
            chans[i] = best_ch
    return ratios, chans


def _summarise_response(ratios: np.ndarray, chans: list, threshold: float) -> dict:
    """Turn per-event response ratios into pct / typical-intensity / dominant-channel.

    'intensity' is the MEDIAN strongest-channel ratio (robust to the occasional huge
    spike), interpretable as "the typical accent is marked at N× a normal burst".
    """
    valid = ~np.isnan(ratios)
    if valid.sum() == 0:
        return {"pct": 0.0, "intensity": 0.0, "dominant": "none"}
    from collections import Counter
    r = ratios[valid]
    notable_idx = [i for i in range(len(ratios))
                   if not np.isnan(ratios[i]) and ratios[i] >= threshold]
    pool = [chans[i] for i in notable_idx] or [chans[i] for i in range(len(ratios))
                                               if chans[i] is not None]
    dominant = Counter(pool).most_common(1)[0][0] if pool else "none"
    return {
        "pct":       round(100.0 * sum(r >= threshold) / len(r), 1),
        "intensity": round(float(np.median(r)), 3),
        "dominant":  dominant,
    }


def _accent_response(
    frames: list,
    dancer_ids: list[int],
    fps: float,
    accent_times: np.ndarray,
    beat_dur: float,
) -> dict:
    """
    Multichannel, partnership-aware response to musical accents detected throughout
    the song. Each accent is judged on the dancer's *strongest* channel (feet / chest
    / hands / head), so the many valid ways to express a hit all count.

    Returns per label a/b: accent_response_pct, accent_hit_mean, accent_dominant_channel.
    Partnership (either-partner): accent_covered_pct (either dancer expressed the accent)
    and framing percentages — one partner goes still while the other expresses
    (frame_a_for_b_pct = 'a' quiet while 'b' spikes; a positive, not a miss).
    """
    THRESHOLD = 1.5           # window peak ≥ 1.5× a typical burst = notable accent mark
    QUIET     = 0.8           # window peak < 0.8× a typical burst = unusually still
    window    = beat_dur * 0.6

    out: dict = {}
    if accent_times is None or len(accent_times) == 0:
        for lbl in ("a", "b"):
            out[f"accent_response_pct_{lbl}"]    = 0.0
            out[f"accent_hit_mean_{lbl}"]        = 0.0
            out[f"accent_dominant_channel_{lbl}"] = "none"
        out["accent_covered_pct"] = 0.0
        out["frame_a_for_b_pct"]  = 0.0
        out["frame_b_for_a_pct"]  = 0.0
        return out

    ratio_by_lbl: dict = {}
    for lbl, did in [("a", dancer_ids[0] if dancer_ids else -1),
                     ("b", dancer_ids[1] if len(dancer_ids) > 1 else -1)]:
        if did < 0:
            ratio_by_lbl[lbl] = (np.full(len(accent_times), np.nan), [None] * len(accent_times))
            out[f"accent_response_pct_{lbl}"]    = 0.0
            out[f"accent_hit_mean_{lbl}"]        = 0.0
            out[f"accent_dominant_channel_{lbl}"] = "none"
            continue
        channels = _expression_channels(frames, did, fps)
        ratios, chans = _event_response(channels, accent_times, window)
        ratio_by_lbl[lbl] = (ratios, chans)
        summ = _summarise_response(ratios, chans, THRESHOLD)
        out[f"accent_response_pct_{lbl}"]     = summ["pct"]
        out[f"accent_hit_mean_{lbl}"]         = summ["intensity"]
        out[f"accent_dominant_channel_{lbl}"] = summ["dominant"]

    ra, _ = ratio_by_lbl["a"]
    rb, _ = ratio_by_lbl["b"]
    covered = framed_ab = framed_ba = 0
    n_valid = 0
    for i in range(len(accent_times)):
        a_ok = not np.isnan(ra[i])
        b_ok = not np.isnan(rb[i])
        if not (a_ok or b_ok):
            continue
        n_valid += 1
        a_hit = a_ok and ra[i] >= THRESHOLD
        b_hit = b_ok and rb[i] >= THRESHOLD
        if a_hit or b_hit:
            covered += 1
        if b_hit and a_ok and ra[i] < QUIET:
            framed_ab += 1      # a went still while b expressed
        if a_hit and b_ok and rb[i] < QUIET:
            framed_ba += 1      # b went still while a expressed
    out["accent_covered_pct"] = round(100.0 * covered / max(n_valid, 1), 1)
    out["frame_a_for_b_pct"]  = round(100.0 * framed_ab / max(n_valid, 1), 1)
    out["frame_b_for_a_pct"]  = round(100.0 * framed_ba / max(n_valid, 1), 1)
    return out


def _texture_match(frames: list, dancer_id: int, fps: float,
                   texture_times: np.ndarray, texture_song: np.ndarray) -> float:
    """
    Time-resolved match of movement texture to the song's bouncy↔smooth texture.

    Movement bounciness per window = within-window spread (std) of the body centre's
    vertical speed: bouncy passages bob sharply up/down (high spread), smooth glides
    keep a steady level (low spread). Correlate against the song texture series.
    Positive r = the dancer gets bouncier when the music does and smooths out when it
    does — i.e. movement quality tracks the song.
    """
    if texture_times is None or len(texture_times) < 4 or texture_song is None:
        return 0.0
    t_c, _x_c, y_c = _center_series(frames, dancer_id)
    if len(t_c) < SMOOTH_WINDOW + 2:
        return 0.0
    y_sm = _smooth(y_c)
    vy   = np.abs(np.diff(y_sm)) / (np.diff(t_c) + 1e-6)
    t_v  = (t_c[:-1] + t_c[1:]) / 2.0
    hop  = float(np.mean(np.diff(texture_times)))

    move_tex = np.full(len(texture_times), np.nan)
    for i, wt in enumerate(texture_times):
        mask = (t_v >= wt) & (t_v < wt + hop)
        if mask.sum() >= 3:
            move_tex[i] = float(np.std(vy[mask]))

    song = np.asarray(texture_song)[:len(texture_times)]
    valid = (~np.isnan(move_tex)) & (~np.isnan(song))
    if valid.sum() < 4:
        return 0.0
    mt = move_tex[valid]
    st = song[valid]
    if np.std(mt) < 1e-9 or np.std(st) < 1e-9:
        return 0.0
    r, _ = _pearsonr(st, mt)
    return round(float(r), 3)


def _phrase_response(
    frames: list,
    dancer_id: int,
    fps: float,
    beat_times: np.ndarray,
    tempo_bpm: float,
) -> dict:
    """
    Multichannel activity at 8-bar (32-beat) phrase boundaries — the same
    expression channels used for accents (feet / chest / hands / head), so a phrase
    change marked with a step, body, or arm all count, not just the free hand.

    phrase_count         int
    phrase_response_pct  float  % of boundaries with a notable response (≥ threshold)
    phrase_hit_mean      float  mean strongest-channel response ratio at boundaries
    """
    if len(beat_times) < 32:
        return {"phrase_count": 0, "phrase_response_pct": 0.0, "phrase_hit_mean": 0.0}

    boundaries = beat_times[::32]           # one boundary every 8 bars (4/4 time)
    beat_dur   = 60.0 / max(tempo_bpm, 60.0)

    channels = _expression_channels(frames, dancer_id, fps)
    if not channels:
        return {"phrase_count": len(boundaries), "phrase_response_pct": 0.0, "phrase_hit_mean": 0.0}

    WINDOW    = beat_dur * 2    # ±2 beats around each boundary
    THRESHOLD = 1.5             # window peak ≥ 1.5× a typical burst = notable response

    ratios, chans = _event_response(channels, np.asarray(boundaries), WINDOW)
    summ = _summarise_response(ratios, chans, THRESHOLD)
    return {
        "phrase_count":        int(np.sum(~np.isnan(ratios))),
        "phrase_response_pct": summ["pct"],
        "phrase_hit_mean":     summ["intensity"],
    }


def compute_musicality(
    frames: list,
    dancer_ids: list[int],
    fps: float,
    beat_times: np.ndarray,
    tempo: float,
    leg_metrics: dict | None = None,
    audio: dict | None = None,
) -> dict:
    """
    Musicality dimensions:

    SONG CHARACTER  — what the music itself asks for (from `audio`).
        bounciness (bouncy/percussive ↔ smooth/legato), dynamic range, accent count

    FREE ARM STYLING  — fluidity and body-responsiveness of unconnected arms.
        shoulder→wrist lateral lag, body–arm correlation, mean wrist speed

    TEXTURE  — does movement quality match the music's character?
        texture match: time-resolved corr of movement bounciness vs song texture
        bounce match: rise/fall rhythm vs detected beat frequency
        music–movement energy tracking: Pearson r between audio RMS and dancer speed
        articulated step on-beat % and timing consistency; weight-only syncopation

    MUSICAL EXPRESSION  — how the dancer marks the accents/phrases the song offers,
        through ANY channel (feet / chest / hands / head) and either partner.
        accent response %, accent hit intensity, dominant channel; 8-bar phrase subset;
        partnership accent coverage + framing; WCS 6/8-count pattern fingerprint
    """
    if len(beat_times) < 2:
        return {"error": "insufficient beat data"}

    audio = audio or {}
    rms_times     = audio.get("rms_times")
    rms_vals      = audio.get("rms_vals")
    texture_times = audio.get("texture_times")
    texture_song  = audio.get("texture")
    accent_times  = audio.get("accent_times", np.array([]))

    beat_interval = float(np.median(np.diff(beat_times)))
    half_beat     = beat_interval / 2.0
    beat_hz       = tempo / 60.0
    beat_dur      = 60.0 / max(tempo, 60.0)

    results: dict = {"tempo_bpm": round(tempo, 1), "beat_count": len(beat_times)}

    # ── SONG CHARACTER (from audio) ──────────────────────────────────────────
    results["song_bounciness"]    = audio.get("song_bounciness", 0.0)
    results["song_dynamic_range"] = audio.get("song_dynamic_range", 0.0)
    results["accent_count"]       = audio.get("accent_count", 0)

    for label, did in [("a", dancer_ids[0] if dancer_ids else -1),
                       ("b", dancer_ids[1] if len(dancer_ids) > 1 else -1)]:
        dancer_label = "lead" if label == "a" else "follow"

        step_data: dict = {}
        if leg_metrics and dancer_label in leg_metrics:
            step_data = leg_metrics[dancer_label].get("step_data", {})
        elif did >= 0:
            step_data = detect_step_events(frames, did, fps)

        art_times = step_data.get("articulated", [])
        wo_times  = step_data.get("weight_only",  [])

        # ── FREE ARM STYLING ─────────────────────────────────────────────────
        arm = _arm_styling_metrics(frames, did, fps) if did >= 0 else {}
        results[f"arm_lag_{label}"]       = arm.get("shoulder_wrist_lag_ms", 0.0)
        results[f"arm_body_corr_{label}"] = arm.get("body_arm_correlation",  0.0)
        results[f"wrist_speed_{label}"]   = arm.get("wrist_speed_mean",      0.0)

        # ── TEXTURE ──────────────────────────────────────────────────────────
        rf_hz = (leg_metrics[dancer_label].get("rise_fall_rhythm_hz", 0.0)
                 if leg_metrics and dancer_label in leg_metrics else 0.0)
        if rf_hz > 0 and beat_hz > 0:
            best_diff    = min(abs(rf_hz - k * beat_hz) for k in [0.5, 1.0, 2.0])
            bounce_match = round(max(0.0, 1.0 - best_diff / beat_hz), 3)
        else:
            bounce_match = 0.0
        results[f"bounce_match_{label}"] = bounce_match

        # Time-resolved texture match: movement bounciness vs song texture
        results[f"texture_match_{label}"] = (
            _texture_match(frames, did, fps, texture_times, texture_song)
            if did >= 0 else 0.0
        )

        mm_corr = 0.0
        if rms_times is not None and rms_vals is not None and len(rms_times) and did >= 0:
            t_c, x_c, y_c = _center_series(frames, did)
            if len(t_c) > SMOOTH_WINDOW + 2:
                x_cs  = _smooth(x_c);  y_cs  = _smooth(y_c)
                bspd  = np.sqrt(np.diff(x_cs)**2 + np.diff(y_cs)**2) / (np.diff(t_c) + 1e-6)
                t_bsp = (t_c[:-1] + t_c[1:]) / 2.0
                t0    = max(rms_times[0], t_bsp[0])
                t1    = min(rms_times[-1], t_bsp[-1])
                if t1 - t0 > 2.0:
                    n_pts = min(500, int((t1 - t0) * 10))
                    t_com = np.linspace(t0, t1, n_pts)
                    r, _  = _pearsonr(
                        np.interp(t_com, rms_times, rms_vals),
                        np.interp(t_com, t_bsp,    bspd),
                    )
                    mm_corr = round(float(r), 3)
        results[f"music_move_corr_{label}"] = mm_corr

        art_stats = _timing_stats(art_times, beat_times, half_beat)
        wo_stats  = _timing_stats(wo_times,  beat_times, half_beat)
        results[f"on_beat_pct_{label}"]     = art_stats["on_beat_pct"]
        results[f"timing_ms_{label}"]       = art_stats["consistency_ms"]
        results[f"syncopation_pct_{label}"] = wo_stats["syncopation_pct"]

        # ── PHRASE CHANGES (8-bar subset, multichannel) ──────────────────────
        pr = _phrase_response(frames, did, fps, beat_times, tempo) if did >= 0 else {}
        results[f"phrase_rsp_pct_{label}"]  = pr.get("phrase_response_pct", 0.0)
        results[f"phrase_hit_mean_{label}"] = pr.get("phrase_hit_mean",     0.0)
        if label == "a":
            results["phrase_count"] = pr.get("phrase_count", 0)

    # ── MUSICAL EXPRESSION at accents throughout the song (multichannel) ──────
    results.update(_accent_response(frames, dancer_ids, fps, accent_times, beat_dur))

    # WCS pattern fingerprint — 6-count and 8-count sequences
    all_steps: list[float] = []
    for label, did in [("a", dancer_ids[0] if dancer_ids else -1),
                       ("b", dancer_ids[1] if len(dancer_ids) > 1 else -1)]:
        dancer_label = "lead" if label == "a" else "follow"
        if leg_metrics and dancer_label in leg_metrics:
            all_steps.extend(leg_metrics[dancer_label].get("step_data", {}).get("all", []))
        elif did >= 0:
            all_steps.extend(detect_step_events(frames, did, fps).get("all", []))
    all_steps.sort()

    six_count = eight_count = 0
    tol = beat_interval * 0.4
    for i in range(len(all_steps) - 5):
        if abs(all_steps[i + 5] - all_steps[i] - 3 * beat_interval) <= tol:
            six_count += 1
        if i + 7 < len(all_steps):
            if abs(all_steps[i + 7] - all_steps[i] - 4 * beat_interval) <= tol:
                eight_count += 1

    results["six_count_patterns"]  = six_count
    results["eight_count_patterns"] = eight_count
    return results


# ---------------------------------------------------------------------------
# Camera setup detection
# ---------------------------------------------------------------------------

def detect_camera_setup(frames: list, dancer_ids: list[int]) -> dict:
    """
    Estimate how the camera is positioned relative to the partnership.

    Metrics:
        partnership_axis_deg   float  angle of lead→follow vector from horizontal
                                      ~0° = side-by-side (camera sees profile of slot)
                                      ~90° = one behind the other (camera looks along slot)
        size_ratio             float  lead body height / follow body height
                                      ~1.0 = both at same depth; >1 = lead closer to camera
        view_angle             str    "side-on" / "diagonal" / "end-on"
        camera_elevation       str    "level" / "elevated" / "low-angle"
        notes                  list   human-readable caveats about measurement quality
    """
    if len(dancer_ids) < 2:
        return {"error": "need 2 dancers"}

    id_a, id_b = dancer_ids[0], dancer_ids[1]

    axis_angles  = []
    bh_a_samples = []
    bh_b_samples = []
    head_y_ratios = []  # head y / frame height — estimates camera elevation

    for f in frames:
        if id_a not in f["dancers"] or id_b not in f["dancers"]:
            continue
        kps_a = f["dancers"][id_a]
        kps_b = f["dancers"][id_b]

        c_a = get_center(kps_a)
        c_b = get_center(kps_b)
        if c_a[2] < CONF_MIN or c_b[2] < CONF_MIN:
            continue

        dx = float(c_b[0] - c_a[0])
        dy = float(c_b[1] - c_a[1])
        axis_angles.append(float(np.degrees(np.arctan2(dy, dx))))

        bh_a = body_height(kps_a)
        bh_b = body_height(kps_b)
        if bh_a > 10 and bh_b > 10:
            bh_a_samples.append(bh_a)
            bh_b_samples.append(bh_b)

        # Head y position as fraction of frame — needs frame height
        head_a = kps_a[KP["head"]]
        if head_a[2] >= CONF_MIN:
            head_y_ratios.append(float(head_a[1]))

    if not axis_angles:
        return {"error": "insufficient data"}

    mean_axis = float(np.median(axis_angles))
    abs_axis  = abs(mean_axis)

    # Size ratio: lead / follow apparent height
    size_ratio = 1.0
    if bh_a_samples and bh_b_samples:
        size_ratio = float(np.median(bh_a_samples) / np.median(bh_b_samples))

    # View angle classification
    if abs_axis < 25:
        view_angle = "side-on"          # camera at ~90° to slot; ideal for seeing footwork
    elif abs_axis > 60:
        view_angle = "end-on"           # camera looking along slot; depth movement hidden
    else:
        view_angle = "diagonal"

    # Camera elevation: estimated from how high the dancers' heads sit in the frame.
    # We compare the mean head y to body height — in an elevated shot the heads
    # appear lower (larger y relative to body height).
    elev_note = "unknown"
    if bh_a_samples and head_y_ratios:
        head_y_mean = float(np.median(head_y_ratios))
        bh_mean     = float(np.median(bh_a_samples))
        # ratio of head-y to body-height: ~1–2 = level, >3 = elevated
        rel = head_y_mean / bh_mean
        if rel < 1.5:
            elev_note = "low-angle"
        elif rel < 3.0:
            elev_note = "level"
        else:
            elev_note = "elevated"

    notes = []
    if abs(size_ratio - 1.0) > 0.2:
        closer = "lead" if size_ratio > 1.0 else "follow"
        notes.append(
            f"{closer} appears {abs(size_ratio-1)*100:.0f}% larger — "
            f"likely closer to camera; depth metrics are approximate"
        )
    if view_angle == "end-on":
        notes.append(
            "end-on camera angle: slot movement goes into/out of frame — "
            "lateral sway and slot metrics will be underestimated"
        )
    if view_angle == "diagonal":
        notes.append(
            "diagonal camera angle: slot and sway are mixed — "
            "interpret lateral metrics with caution"
        )

    return {
        "partnership_axis_deg": round(mean_axis, 1),
        "size_ratio":           round(size_ratio, 3),
        "view_angle":           view_angle,
        "camera_elevation":     elev_note,
        "notes":                notes,
    }


# ---------------------------------------------------------------------------
# Top-level aggregation
# ---------------------------------------------------------------------------

def compute_all_metrics(pose_data: dict) -> dict:
    """
    Run all four analyses and return a combined metrics dict.
    Requires 'video_path' key in pose_data for beat extraction.
    """
    frames     = pose_data["frames"]
    fps        = pose_data["fps"]
    ids        = pose_data.get("dancer_ids", [])

    # JSON round-trip converts int dict keys to strings — normalise back to int
    # Also re-hydrate keypoint arrays (stored as nested lists or numpy-string repr)
    for f in frames:
        if "dancers" not in f:
            continue
        d = f["dancers"]
        if not d:
            continue
        # re-key from str → int if needed
        if isinstance(next(iter(d)), str):
            f["dancers"] = {int(k): v for k, v in d.items()}
        # convert each keypoint value to ndarray
        for did, kps in list(f["dancers"].items()):
            if isinstance(kps, np.ndarray):
                continue
            if isinstance(kps, str):
                # numpy str repr e.g. "[[x y c]\n ...]" → parse with fromstring trick
                import re as _re
                nums = [float(x) for x in _re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", kps)]
                f["dancers"][did] = np.array(nums, dtype=float).reshape(-1, 3)
            else:
                f["dancers"][did] = np.array(kps, dtype=float)
    video_path = pose_data.get("video_path", "")

    audio: dict = {}
    if video_path:
        try:
            audio = extract_audio_features(video_path)
        except Exception as e:
            print(f"  [!] Audio analysis failed: {e}")

    metrics = {}

    metrics["camera_setup"] = detect_camera_setup(frames, ids[:2])

    total_frames = len(frames)
    metrics["tracking_quality"] = {}
    for i, did in enumerate(ids[:2]):
        label = "lead" if i == 0 else "follow"
        n_tracked = sum(1 for f in frames if did in f["dancers"])
        pct = round(100.0 * n_tracked / max(total_frames, 1), 1)
        metrics["tracking_quality"][label] = {
            "frames_tracked": n_tracked,
            "total_frames":   total_frames,
            "coverage_pct":   pct,
        }

    for i, did in enumerate(ids[:2]):
        label = "lead" if i == 0 else "follow"
        metrics[f"leg_action_{label}"]  = compute_leg_action(frames, did, fps)
        metrics[f"body_action_{label}"] = compute_body_action(frames, did, fps)

    slot_axis, _ = _slot_axis(frames, ids[:2])
    metrics["weight_countering"] = compute_weight_countering(frames, ids[:2], fps, slot_axis)

    # Travel decomposition: couple-around-room, down-the-slot (per dancer),
    # and (via the post metrics in weight_countering) stretch/compression.
    travel = compute_travel(frames, ids[:2], fps, slot_axis)
    metrics["travel"] = {
        "slot_axis_deg":          travel["slot_axis_deg"],
        "couple_travel_path_bh":  travel["couple_travel_path_bh"],
        "couple_travel_range_bh": travel["couple_travel_range_bh"],
    }
    metrics["travel_lead"]   = travel["lead"]
    metrics["travel_follow"] = travel["follow"]

    # Pass pre-computed leg metrics so musicality doesn't re-run step detection
    leg_for_music = {
        "lead":   metrics.get("leg_action_lead",   {}),
        "follow": metrics.get("leg_action_follow",  {}),
    }
    metrics["musicality"] = compute_musicality(
        frames, ids[:2], fps,
        audio.get("beat_times", np.array([])),
        audio.get("tempo", 0.0),
        leg_metrics=leg_for_music,
        audio=audio,
    )

    # Always-on deep detail (per-step distributions, accent-vs-anchor timing, coupling)
    tempo_v  = float(audio.get("tempo", 0.0) or 0.0)
    beat_dur = 60.0 / tempo_v if tempo_v > 0 else 0.0
    accents  = audio.get("accent_times", np.array([]))
    metrics["movement_quality_detail"] = {}
    for i, did in enumerate(ids[:2]):
        label = "lead" if i == 0 else "follow"
        step_data = metrics.get(f"leg_action_{label}", {}).get("step_data", {})
        metrics["movement_quality_detail"][label] = compute_articulation_detail(
            frames, did, fps, step_data, accents, beat_dur,
        )

    return metrics
