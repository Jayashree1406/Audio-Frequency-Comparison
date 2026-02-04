import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load audio files
audio1, sr1 = librosa.load("audio21.mpeg", sr=None)
audio2, sr2 = librosa.load("audio22.mpeg", sr=None)

# Match sampling rate
sr = min(sr1, sr2)
audio1 = librosa.resample(audio1, orig_sr=sr1, target_sr=sr)
audio2 = librosa.resample(audio2, orig_sr=sr2, target_sr=sr)

# Trim to same length
min_len = min(len(audio1), len(audio2))
audio1 = audio1[:min_len]
audio2 = audio2[:min_len]

# FFT (frequency domain)
fft1 = np.abs(np.fft.fft(audio1))
fft2 = np.abs(np.fft.fft(audio2))

# Use only positive frequencies
fft1 = fft1[:len(fft1)//2]
fft2 = fft2[:len(fft2)//2]

# Similarity score (cosine similarity)
similarity_score = cosine_similarity(
    fft1.reshape(1, -1),
    fft2.reshape(1, -1)
)[0][0]

print("Similarity Score:", similarity_score)

# Plot frequency-domain graph
plt.figure(figsize=(10, 5))
plt.plot(fft1, label="Audio 1", alpha=0.8)
plt.plot(fft2, label="Audio 2", alpha=0.8)
plt.title("Frequency Domain Comparison")
plt.xlabel("Frequency Bins")
plt.ylabel("Magnitude")
plt.text(
    0.6 * len(fft1),
    max(max(fft1), max(fft2)) * 0.8,
    f"Similarity Score: {similarity_score:.3f}",
    fontsize=12,
    bbox=dict(facecolor='white', alpha=0.7)
)
plt.legend()
plt.grid(True)
plt.show()