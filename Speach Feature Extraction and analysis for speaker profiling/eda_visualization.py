import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

# Set the folder where your audio files are
audio_folder = "audio"
output_folder = "plots"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)

# List all .wav files
files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

# Loop through each file
for file in files:
    file_path = os.path.join(audio_folder, file)
    print(f"🎧 Processing {file}...")

    y, sr = librosa.load(file_path)
    base_name = os.path.splitext(file)[0]

    # Plot 1: Waveform
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title(f"Waveform - {file}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{base_name}_waveform.png")
    plt.close()

    # Plot 2: MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title(f"MFCCs - {file}")
    plt.ylabel("MFCC Coefficients")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{base_name}_mfcc.png")
    plt.close()

    # Plot 3: Pitch (F0)
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=300)
    plt.figure(figsize=(10, 4))
    plt.plot(f0, color='green')
    plt.title(f"Pitch Contour (F0) - {file}")
    plt.xlabel("Frame")
    plt.ylabel("Pitch (Hz)")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{base_name}_pitch.png")
    plt.close()

    # Plot 4: RMS Energy
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(range(len(rms)), sr=sr)
    plt.figure(figsize=(10, 4))
    plt.plot(times, rms, color='orange')
    plt.title(f"Energy (RMS) - {file}")
    plt.xlabel("Time (s)")
    plt.ylabel("RMS Energy")
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{base_name}_energy.png")
    plt.close()

    print(f" Plots saved for {file} in /{output_folder}")

print("\n All visualizations saved in the 'plots/' folder!")
