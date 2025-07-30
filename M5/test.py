import pydub
import numpy as np
import matplotlib.pyplot as plt

OUTPUT = "clip44_roundtrip.mp3"


def read(f, normalized=False):
    a = pydub.AudioSegment.from_mp3(f)
    y = np.array(a.get_array_of_samples())
    
    if a.channels == 2:
        y = y.reshape((-1, 2))
    

    if normalized:
        return a.frame_rate, np.float32(y) / (2**15)
    else:
        return a.frame_rate, y


def write(f, sr, x, normalized=False):
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    
    if normalized:
        # normalized array — each item should be a float in [−1, 1)
        y = np.int16(x * 2**15)
    else:
        y = np.int16(x)

    song = pydub.AudioSegment(
        y.tobytes(),
        frame_rate=sr,
        sample_width=2,
        channels=channels
    )

    song.export(f, format="mp3", bitrate="320k")


def plot_audio(file_path, normalized=False):
    # 1) Load
    sr, x = read(file_path, normalized=normalized)

    # 3) Time axis
    t = np.arange(len(x)) / sr

    # 4) Plot waveform
    plt.figure(figsize=(10, 3))
    plt.plot(t, x)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Waveform of {file_path}')
    plt.tight_layout()
    plt.show()


def filterbank(sig, sr):

    bandlimits = [0, 200, 400, 800, 1600, 3200]
    band_number = len(bandlimits)

    n = len(sig)
    # full FFT (complex spectrum length n, symmetric)
    fhat = np.fft.fft(sig)
    
    # frequency axis for each bin, from -sr/2..+sr/2
    freqs = np.fft.fftfreq(n, d=1/sr)
    nyquist = sr/2
    

    band_fft = np.zeros((n, band_number), dtype=complex)
    
    # for each band, zero out bins outside [low, high)
    for i, low in enumerate(bandlimits):
        high = bandlimits[i+1] if i+1 < band_number else nyquist
        # mask bins whose absolute frequency is in the band
        mask = (np.abs(freqs) >= low) & (np.abs(freqs) < high)
        band_fft[mask, i] = fhat[mask]
    
    return band_fft


# 1) Read the original into float32 in [–1,1)
sr, x = read('M5/clip.mp3', normalized=True)
print(f"→ Read  “{'clip44.mp3'}”:  sr={sr}, shape={x.shape}, dtype={x.dtype}")
print(x)
print(20 * '-')

# 2) Write it back out
write(OUTPUT, sr, x, normalized=True)
print(f"→ Wrote “{OUTPUT}”")

print(filterbank(x, sr))

plot_audio('clip.mp3')
