import numpy as np
import librosa
from pathlib import Path

CONTRAST_DB = 6    # how much the call band must stand out (raise = stricter)
MIN_DUR = 0.1      # ignore detections shorter than this (seconds)
MERGE_GAP = 0.5    # join detections closer than this (seconds)
PAD = 0.25         # extra time kept around each call (seconds)
FRAME_S = 0.05     # time resolution (seconds)

Path("output").mkdir(exist_ok=True)

for wav in sorted(Path("recordings").glob("*.wav")):
    y, sr = librosa.load(wav, sr=22050, mono=True)
    hop = int(FRAME_S * sr)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    def band(lo, hi):
        m = (freqs >= lo) & (freqs <= hi)
        return 10 * np.log10(S[m].mean(axis=0) + 1e-12)

    # loudness of the call band minus loudness of the bands beside it
    contrast = band(1600, 2500) - np.maximum(band(800, 1400), band(2800, 3600))
    excess = contrast - np.percentile(contrast, 25)
    active = excess > CONTRAST_DB

    # find where each active stretch starts and ends (in seconds)
    edges = np.diff(np.concatenate([[0], active.astype(int), [0]]))
    starts = np.where(edges == 1)[0] * FRAME_S
    ends = np.where(edges == -1)[0] * FRAME_S

    # merge nearby stretches
    calls = []
    for s, e in zip(starts, ends):
        if calls and s - calls[-1][1] <= MERGE_GAP:
            calls[-1][1] = e
        else:
            calls.append([s, e])

    # drop very short ones and add padding
    calls = [(max(0, s - PAD), e + PAD) for s, e in calls if e - s >= MIN_DUR]

    # save Audacity labels
    with open(f"output/{wav.stem}.labels.txt", "w") as f:
        for i, (s, e) in enumerate(calls):
            f.write(f"{s:.3f}\t{e:.3f}\tcall_{i}\n")

    print(f"{wav.name}: {len(calls)} calls, strongest peak {excess.max():.1f} dB")
