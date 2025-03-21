import librosa
import numpy as np
import pandas as pd
import os

def extract_features_from_csv(csv_path, output_dir="features/"):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.")
        return

    os.makedirs(output_dir, exist_ok=True)

    data = pd.read_csv(csv_path)

    # Create empty dataframes for each feature
    mfcc_df = pd.DataFrame()
    chroma_df = pd.DataFrame()
    mel_spec_df = pd.DataFrame()
    contrast_df = pd.DataFrame()
    tonnetz_df = pd.DataFrame()
    zcr_df = pd.DataFrame()
    rmse_df = pd.DataFrame()

    for idx, row in data.iterrows():
        audio_file = row['audio_file']
        emotion = row['emotion']

        if not os.path.exists(audio_file):
            print(f"Warning: Audio file '{audio_file}' not found. Skipping...")
            continue

        try:
            audio, sr = librosa.load(audio_file, sr=None)
            print(f"Loaded audio file '{audio_file}' with sample rate {sr}.")

            # Feature 1: MFCCs
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_df = pd.concat([mfcc_df, pd.DataFrame([mfcc_mean], columns=[f"mfcc_{i+1}" for i in range(13)])], ignore_index=True)

            # Feature 2: Chroma
            stft = np.abs(librosa.stft(audio))
            chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            chroma_df = pd.concat([chroma_df, pd.DataFrame([chroma_mean], columns=[f"chroma_{i+1}" for i in range(12)])], ignore_index=True)

            # Feature 3: Mel-Spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
            mel_mean = np.mean(mel_spec, axis=1)
            mel_spec_df = pd.concat([mel_spec_df, pd.DataFrame([mel_mean], columns=[f"mel_{i+1}" for i in range(len(mel_mean))])], ignore_index=True)

            # Feature 4: Spectral Contrast
            contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            contrast_df = pd.concat([contrast_df, pd.DataFrame([contrast_mean], columns=[f"contrast_{i+1}" for i in range(contrast.shape[0])])], ignore_index=True)

            # Feature 5: Tonnetz
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
            tonnetz_mean = np.mean(tonnetz, axis=1)
            tonnetz_df = pd.concat([tonnetz_df, pd.DataFrame([tonnetz_mean], columns=[f"tonnetz_{i+1}" for i in range(tonnetz.shape[0])])], ignore_index=True)

            # Feature 6: Zero Crossing Rate (ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)
            zcr_mean = np.mean(zcr)
            zcr_df = pd.concat([zcr_df, pd.DataFrame([[zcr_mean]], columns=["zcr_mean"])], ignore_index=True)

            # Feature 7: Root Mean Square Energy (RMSE)
            rmse = librosa.feature.rms(y=audio)
            rmse_mean = np.mean(rmse)
            rmse_df = pd.concat([rmse_df, pd.DataFrame([[rmse_mean]], columns=["rmse_mean"])], ignore_index=True)

        except Exception as e:
            print(f"Error processing '{audio_file}': {e}")

    # Add emotion column to each feature dataframe
    mfcc_df['emotion'] = data['emotion']
    chroma_df['emotion'] = data['emotion']
    mel_spec_df['emotion'] = data['emotion']
    contrast_df['emotion'] = data['emotion']
    tonnetz_df['emotion'] = data['emotion']
    zcr_df['emotion'] = data['emotion']
    rmse_df['emotion'] = data['emotion']

    # Save each feature-specific dataframe to a separate CSV
    mfcc_df.to_csv(os.path.join(output_dir, "mfcc_features.csv"), index=False)
    chroma_df.to_csv(os.path.join(output_dir, "chroma_features.csv"), index=False)
    mel_spec_df.to_csv(os.path.join(output_dir, "mel_spectrogram_features.csv"), index=False)
    contrast_df.to_csv(os.path.join(output_dir, "spectral_contrast_features.csv"), index=False)
    tonnetz_df.to_csv(os.path.join(output_dir, "tonnetz_features.csv"), index=False)
    zcr_df.to_csv(os.path.join(output_dir, "zcr_features.csv"), index=False)
    rmse_df.to_csv(os.path.join(output_dir, "rmse_features.csv"), index=False)

    print(f"\nFeature extraction complete! Feature CSVs saved in '{os.path.abspath(output_dir)}' directory.")

csv_path = "audio_data.csv" 
extract_features_from_csv(csv_path)
