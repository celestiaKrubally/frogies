import time, warnings
import numpy as np, pandas as pd
import librosa, soundfile as sf
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# ================= PATHS =================
IN_DIR  = Path("/content/drive/MyDrive/frogie_project/recordings")
OUT_DIR = Path("/content/drive/MyDrive/frogie_project/output")

# ================= SETTINGS (tune these) =================
LOW_HZ, HIGH_HZ = 1500, 2600   # call band (a bit wider than 1600-2500)
SR = 22050                     # sample rate used for analysis
FRAME_S = 0.05                 # time resolution (s)
ABOVE_NOISE_DB = 8             # dB above noise floor that counts as a call (lower = more sensitive)
NOISE_PERCENTILE = 20          # noise floor = this percentile of band level (robust in loud choruses)
ABS_MIN_DB = -65               # absolute floor so silent files give nothing
MIN_DUR = 0.1                  # drop detections shorter than this (s)
MERGE_GAP = 0.5                # join detections closer than this (s)
PAD = 0.25                     # padding around each detection (s)

REJECT_CLICKS = True           # ignore frames with a broadband burst (clicks, bumps, drips)
CLICK_LOW_HZ, CLICK_HIGH_HZ = 6000, 10000
CLICK_DB = 12                  # high band this far above its own noise = click (raise = less strict)

WRITE_TRIMMED = False          # True = also save a _trimmed.wav per file (big for long files)
MAKE_PLOTS = True              # save a spectrogram + detection plot per file
PLOT_SECONDS = 60              # length of the plotted window (starts at the first detection)
PLOT_MAX_HZ = 5000

# ================= SETUP =================
if not Path("/content/drive/MyDrive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

OUT_DIR.mkdir(parents=True, exist_ok=True)


def band_db(y, sr, lo, hi, hop):
    """Level (dB) of a frequency band, one value per frame."""
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    rms = librosa.feature.rms(y=sosfiltfilt(sos, y), frame_length=hop * 2, hop_length=hop)[0]
    return 20 * np.log10(np.maximum(rms, 1e-10))


def detect(path):
    y, sr = librosa.load(path, sr=SR, mono=True)
    hop = int(FRAME_S * sr)
    db = band_db(y, sr, LOW_HZ, HIGH_HZ, hop)

    nf = np.percentile(db, NOISE_PERCENTILE)
    th = max(nf + ABOVE_NOISE_DB, ABS_MIN_DB)
    active = db > th

    n_click = 0
    if REJECT_CLICKS:
        hi = band_db(y, sr, CLICK_LOW_HZ, CLICK_HIGH_HZ, hop)
        click = (hi - np.percentile(hi, NOISE_PERCENTILE)) > CLICK_DB
        n_click = int((active & click).sum())
        active = active & ~click

    # active stretches -> (start_frame, end_frame)
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]

    merged = []
    for s, e in zip(starts, ends):
        if merged and (s - merged[-1][1]) * FRAME_S <= MERGE_GAP:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # per-call measurements
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bm = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)

    calls = []
    for s, e in merged:
        if (e - s) * FRAME_S < MIN_DUR:
            continue
        spec = S[bm, s:e].mean(axis=1) if e > s else None
        calls.append(dict(
            start=max(0.0, s * FRAME_S - PAD),
            end=min(len(y) / sr, e * FRAME_S + PAD),
            peak_db=float(db[s:e].max()),
            mean_db=float(db[s:e].mean()),
            peak_hz=float(freqs[bm][spec.argmax()]) if spec is not None and spec.size else np.nan,
        ))
    return dict(y=y, sr=sr, dur=len(y) / sr, db=db, S=S, freqs=freqs,
                nf=nf, th=th, maxdb=float(db.max()), n_click=n_click, calls=calls)


def make_plot(r, calls, tag):
    t0 = max(0.0, (calls[0]["start"] - 2) if calls else 0.0)
    f0 = int(t0 / FRAME_S)
    f1 = min(r["S"].shape[1], f0 + int(PLOT_SECONDS / FRAME_S))
    t0, t1 = f0 * FRAME_S, f1 * FRAME_S
    fm = r["freqs"] <= PLOT_MAX_HZ

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1.2]})
    img = librosa.amplitude_to_db(r["S"][fm, f0:f1], ref=np.max)
    a1.imshow(img, origin="lower", aspect="auto", cmap="magma",
              extent=[t0, t1, r["freqs"][fm][0], r["freqs"][fm][-1]])
    for hz in (LOW_HZ, HIGH_HZ):
        a1.axhline(hz, color="white", ls="--", lw=0.6, alpha=0.6)
    for c in calls:
        if c["end"] >= t0 and c["start"] <= t1:
            a1.axvspan(c["start"], c["end"], color="cyan", alpha=0.2)
    a1.set_ylabel("Hz")
    a1.set_title(f"{tag}  |  {len(calls)} detections (cyan)  |  window {t0:.0f}-{t1:.0f}s")

    t = np.arange(f0, f1) * FRAME_S
    a2.plot(t, r["db"][f0:f1], lw=0.8)
    a2.axhline(r["th"], color="red", ls="--", lw=1, label="threshold")
    a2.axhline(r["nf"], color="gray", ls=":", lw=1, label="noise floor")
    a2.set_ylabel("band dB"); a2.set_xlabel("seconds"); a2.legend(loc="upper right")
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"{tag}.png", dpi=110)
    plt.show(); plt.close(fig)


# ================= FIND FILES =================
wavs = sorted(p for p in IN_DIR.rglob("*")
              if p.suffix.lower() == ".wav" and not p.name.startswith("."))
print(f"Looking in: {IN_DIR}")
print(f"Found {len(wavs)} wav file(s):")
for w in wavs:
    print("  -", w.relative_to(IN_DIR))
if not wavs:
    raise SystemExit("No wav files found. Check IN_DIR.")

# ================= RUN =================
summary, all_calls = [], []
for n, wav in enumerate(wavs, 1):
    tag = "_".join(wav.relative_to(IN_DIR).with_suffix("").parts)
    t_start = time.time()
    print(f"\n[{n}/{len(wavs)}] {wav.name}")
    try:
        r = detect(wav)
    except Exception as ex:
        print(f"  ERROR, skipped: {ex}")
        summary.append([wav.name, np.nan, "error", np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
        continue

    calls = r["calls"]
    kept = sum(c["end"] - c["start"] for c in calls)
    per_min = len(calls) / (r["dur"] / 60) if r["dur"] else 0

    print(f"  length: {r['dur']/60:.1f} min | noise floor: {r['nf']:.1f} dB | threshold: {r['th']:.1f} dB")
    print(f"  loudest frame: {r['maxdb']:.1f} dB ({r['maxdb'] - r['th']:+.1f} dB vs threshold)")
    print(f"  detections: {len(calls)} | in calls: {kept:.0f}s ({100*kept/r['dur']:.1f}%) | {per_min:.1f} calls/min")
    if REJECT_CLICKS:
        print(f"  click frames rejected: {r['n_click']}")
    if calls:
        hz = np.nanmedian([c["peak_hz"] for c in calls])
        print(f"  median peak frequency: {hz:.0f} Hz")
        print("  first calls (s): " + ", ".join(f"{c['start']:.1f}-{c['end']:.1f}" for c in calls[:8])
              + (" ..." if len(calls) > 8 else ""))
    else:
        gap = r["th"] - r["maxdb"]
        print(f"  0 detections: loudest frame is {gap:.1f} dB under the threshold.")
        print(f"  Try ABOVE_NOISE_DB = {max(1, ABOVE_NOISE_DB - gap - 1):.0f}, or check the plot / Audacity.")

    with open(OUT_DIR / f"{tag}.labels.txt", "w") as f:
        for i, c in enumerate(calls):
            f.write(f"{c['start']:.3f}\t{c['end']:.3f}\tcall_{i}\n")

    if WRITE_TRIMMED and calls:
        sf.write(OUT_DIR / f"{tag}_trimmed.wav",
                 np.concatenate([r["y"][int(c["start"] * r["sr"]):int(c["end"] * r["sr"])] for c in calls]),
                 r["sr"])

    for i, c in enumerate(calls):
        all_calls.append([wav.name, i, round(c["start"], 3), round(c["end"], 3),
                          round(c["end"] - c["start"], 3), round(c["peak_db"], 1),
                          round(c["mean_db"], 1), round(c["peak_hz"], 0)])
    summary.append([wav.name, round(r["dur"], 1), len(calls), round(kept, 1), round(per_min, 2),
                    round(r["nf"], 1), round(r["th"], 1), round(r["maxdb"], 1), r["n_click"]])

    if MAKE_PLOTS:
        make_plot(r, calls, tag)
    print(f"  done in {time.time() - t_start:.0f}s")

# ================= SAVE + SHOW =================
df = pd.DataFrame(summary, columns=["file", "length_s", "n_detections", "seconds_kept", "calls_per_min",
                                    "noise_floor_db", "threshold_db", "loudest_db", "click_frames"])
df.to_csv(OUT_DIR / "summary.csv", index=False)
pd.DataFrame(all_calls, columns=["file", "call_id", "start_s", "end_s", "dur_s",
                                 "peak_db", "mean_db", "peak_hz"]).to_csv(OUT_DIR / "calls.csv", index=False)

print("\n=== SUMMARY ===")
display(df)
print(f"Saved labels, plots, summary.csv and calls.csv to {OUT_DIR}")