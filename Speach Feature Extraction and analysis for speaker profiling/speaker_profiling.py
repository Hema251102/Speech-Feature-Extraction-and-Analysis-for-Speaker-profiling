#  Import necessary libraries
import os
import librosa
import numpy as np
import pandas as pd
import speech_recognition as sr

#  Set the folder path (relative path since VS Code is opened here)
audio_folder = "audio"
output_csv = "output.csv"

#  Get all .wav files from the folder
files = [f for f in os.listdir(audio_folder) if f.endswith(".wav")]

#  List to store results
data = []

#  Initialize recognizer once
recognizer = sr.Recognizer()

#  Loop through each audio file
for file in files:
    file_path = os.path.join(audio_folder, file)

    #  Load audio file using librosa
    y, sr_audio = librosa.load(file_path)

    #  Duration
    duration = len(y) / sr_audio

    #  Energy (Root Mean Square)
    rms = librosa.feature.rms(y=y)[0]
    energy = np.mean(rms)

    #  Pitch using pyin
    f0, _, _ = librosa.pyin(y, fmin=75, fmax=300)
    f0 = f0[~np.isnan(f0)]
    pitch = np.mean(f0) if len(f0) > 0 else 0

    #  MFCC (take first 5 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr_audio, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)

    #  Speech-to-Text using SpeechRecognition
    try:
        with sr.AudioFile(file_path) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        text = "Could not understand"
    except sr.RequestError:
        text = "API unavailable"
    except:
        text = ""

    #  Store results
    data.append({
        "File Name": file,
        "Duration (sec)": round(duration, 2),
        "Pitch (Hz)": round(pitch, 2),
        "Energy (RMS)": round(energy, 4),
        "MFCC_1": round(mfcc_mean[0], 2),
        "MFCC_2": round(mfcc_mean[1], 2),
        "MFCC_3": round(mfcc_mean[2], 2),
        "MFCC_4": round(mfcc_mean[3], 2),
        "MFCC_5": round(mfcc_mean[4], 2),
        "Transcription": text
    })

#  Save results to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)

print("✅ Feature extraction complete. Saved to:", output_csv)
