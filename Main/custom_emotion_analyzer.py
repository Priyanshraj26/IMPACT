import librosa
import numpy as np
import pandas as pd
from collections import Counter
from tensorflow.keras.models import model_from_json

def analyze_emotion_from_audio(audio_path, model_json_path, model_weights_path,
                                segment_length=4, stride=2, rmse_threshold=0.01):
    emotion_labels = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    emotion_mapping = {
        'happy': 'happy',
        'neutral': 'neutral',
        'angry': 'nervous',
        'fear': 'nervous',
        'sad': 'nervous',
        'surprise': 'nervous'
    }

    y, sr = librosa.load(audio_path, sr=None)
    samples_per_segment = int(segment_length * sr)
    stride_samples = int(stride * sr)

    with open(model_json_path, 'r') as f:
        model = model_from_json(f.read())
    model.load_weights(model_weights_path)

    def extract_features(segment, sr):
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc = mfcc.T
        features = np.mean(mfcc, axis=1, keepdims=True)
        return features

    def is_silent(segment, threshold=rmse_threshold):
        return np.sqrt(np.mean(segment**2)) < threshold

    emotion_counts = Counter()
    start = 0
    total_segments = 0
    valid_segments = 0

    while start + samples_per_segment <= len(y):
        segment = y[start:start + samples_per_segment]
        total_segments += 1

        if not is_silent(segment):
            valid_segments += 1
            features = extract_features(segment, sr)
            features = np.expand_dims(features, axis=0)
            prediction = model.predict(features, verbose=0)
            predicted_emotion = emotion_labels[np.argmax(prediction)]
            mapped_emotion = emotion_mapping.get(predicted_emotion, 'nervous')
            emotion_counts[mapped_emotion] += 1

        start += stride_samples

    df = pd.DataFrame.from_dict(emotion_counts, orient='index', columns=['Segments Count'])
    df.index.name = 'Emotion'
    df = df.reset_index()

    if emotion_counts:
        final_emotion = emotion_counts.most_common(1)[0][0]
        df.loc[len(df.index)] = ['Final Emotion', final_emotion]
    else:
        final_emotion = "No valid segments"
        df.loc[len(df.index)] = ['Final Emotion', final_emotion]

    stats = {
        "total_segments": total_segments,
        "valid_segments": valid_segments
    }

    return df, final_emotion, stats
