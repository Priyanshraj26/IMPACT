import librosa
import whisper

whisper_model = whisper.load_model("medium")

def transcribe_audio(audio_path):
    """
    Transcribes audio using Whisper and returns the text.
    """
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        return f"Error in transcription: {e}"

def calculate_wpm(audio_path, transcript):
    """
    Calculates words per minute and returns WPM, category, and duration.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration_seconds = len(y) / sr
        word_count = len(transcript.split())
        wpm = round(word_count / (duration_seconds / 60), 2)

        if wpm < 100:
            category = "Slow"
        elif 100 <= wpm <= 150:
            category = "Normal"
        else:
            category = "Fast"

        return wpm, category, duration_seconds
    except Exception as e:
        return f"Error in WPM calculation: {e}", None, None
