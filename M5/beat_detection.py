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


def plot_audio(sr, x, normalized=False):
    # Load


    if x.ndim == 2:
        x = x[:,0]

    x_pos = np.clip(x, 0, None)

    t = np.arange(len(x_pos)) / sr

    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(t, x_pos)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (only ≥ 0)')
    plt.title('Waveform')
    plt.tight_layout()
    plt.show()


def detect_beats(sr, x):
    pass



# 1) Read the original into float32 in [–1,1)
sr, x = read('clip.mp3', normalized=True)
print(f"→ Read  “{'clip44.mp3'}”:  sr={sr}, shape={x.shape}, dtype={x.dtype}")
print(x)
print(20 * '-')

# 2) Write it back out
write(OUTPUT, sr, x, normalized=True)
print(f"→ Wrote “{OUTPUT}”")

plot_audio(sr, x)
detect_beats(sr, x)