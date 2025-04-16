import streamlit as st
import nltk
import whisper
import language_tool_python
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import librosa
from deepface import DeepFace
from moviepy import VideoFileClip
import cv2
from custom_emotion_analyzer import analyze_emotion_from_audio
from dynamic_suggestions import get_grammar_suggestions

# Download necessary NLTK data
nltk.download('punkt')

# Load Whisper model
whisper_model = whisper.load_model("medium")

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app
st.title("Speech & Video Emotion & Analysis")

# Upload audio or video file
uploaded_file = st.file_uploader("Upload an audio/video file", type=["wav", "mp4", "avi", "mov"])

if uploaded_file is not None:
    file_extension = uploaded_file.name.split(".")[-1]
    
    # **CASE 1: VIDEO FILE PROCESSING**
    if file_extension in ["mp4", "avi", "mov"]:
        video_path = f"temp_video.{file_extension}"

        # Save uploaded video file
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Ensure file is saved
        if not os.path.exists(video_path):
            st.error("Error: Video file not saved properly.")
        else:

            # Extract frames for facial emotion detection
            st.write("Extracting frames...")
            cap = cv2.VideoCapture(video_path)
            frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
            os.makedirs("frames", exist_ok=True)
            frame_count = 0
            frame_sample_rate = max(1, frame_rate // 2)  # Capture every half-second frame
            emotions = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_sample_rate == 0:
                    frame_path = f"frames/frame_{frame_count}.jpg"
                    cv2.imwrite(frame_path, frame)
                    analysis = DeepFace.analyze(img_path=frame_path, actions=['emotion'], enforce_detection=False)
                    emotions.append(analysis[0]['dominant_emotion'])
                frame_count += 1
            cap.release()

            emotion_mapping = {
                                "happy": "Happy",
                                "neutral": "Neutral",
                            }
            
            dominant_facial_emotion = max(set(emotions), key=emotions.count) if emotions else "No Face Detected"
            dominant_facial_emotion = emotion_mapping.get(dominant_facial_emotion, "Nervous")  # Default to "Nervous" for others

            # Extract audio from video
            st.write("Extracting audio...")
            audio_path = "audio.wav"
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(audio_path)
            video.close()

    else:
        # **CASE 2: AUDIO FILE PROCESSING**
        audio_path = "temp_audio.wav"
        with open(audio_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # **Speech Processing (Common for Audio & Video)**
    st.write("Processing speech...")

    # Load audio and transcribe
    y, sr = librosa.load(audio_path, sr=16000)
    text = whisper_model.transcribe(audio_path)["text"]
    duration_seconds = len(y) / sr

    # WPM Calculation
    def classify_wpm(wpm):
        if wpm < 100:
            return "Slow"
        elif 100 <= wpm <= 150:
            return "Normal"
        else:
            return "Fast"
    
    wpm = round(len(text.split()) / (duration_seconds / 60), 2)

    wpm_category = classify_wpm(wpm)

    # Grammar Checking
    tool = language_tool_python.LanguageTool('en-US')
    sentences = sent_tokenize(text)
    grammar_errors = sum(len(tool.check(sentence)) for sentence in sentences)
    final_grammar_score = max(0, 10 - ((grammar_errors / max(len(text.split()), 1)) * 100))

    # Sentiment Analysis
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    sentiment = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"

    # Define original emotion labels from the model
    speech_emotions_map = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgusted", "Surprised"]

    #  Emotion grouping for speech (same as facial emotions)
    speech_emotion_mapping = {
        "Happy": "Happy",
        "Neutral": "Neutral",
        "Calm": "Neutral",  # Treat "Calm" as "Neutral"
        "Angry": "Nervous",
        "Fearful": "Nervous",
        "Sad": "Nervous",
        "Disgusted": "Nervous",
    }

    # Emotion Analysis using DeepFace
    if file_extension in ["mp4", "avi", "mov"]:
        st.write("### 📷 Facial Emotion:", dominant_facial_emotion)

    st.write("### 🔍 Segment-based Emotion Analysis (Your Model)")

    model_json_path = "CNN_model.json"
    model_weights_path = "best_model.keras"

    if os.path.exists(model_json_path) and os.path.exists(model_weights_path):
        df_custom_emotion, final_custom_emotion, stats = analyze_emotion_from_audio(
            audio_path=audio_path,
            model_json_path=model_json_path,
            model_weights_path=model_weights_path
        )
        st.dataframe(df_custom_emotion)
        st.write(f"✅ Final Emotion (Segment-wise): **{final_custom_emotion}**")
        st.write(f"Segments processed: {stats['valid_segments']} / {stats['total_segments']}")
    else:
        st.warning("Custom model files not found. Please upload 'model_architecture.json' and 'model_weights.keras'.")


    # Show Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h2>Words Per Minute (WPM)</h2>", unsafe_allow_html=True)
        st.write(f"**Words Per Minute (WPM):** {wpm} ({wpm_category})")
    with col2:
        st.markdown("<h2>Sentiment Analysis</h2>", unsafe_allow_html=True)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Compound Score: {compound_score}")
    with col3:
        st.markdown("<h2>Grammar Score</h2>", unsafe_allow_html=True)
        st.write(f"{final_grammar_score}/10")

    # Grammar Suggestions
    st.write("###  Grammar Suggestions")
    with st.expander("Fluency Suggestions"):
        suggestions = get_grammar_suggestions(final_grammar_score)
        st.text(suggestions)

    with st.expander("Transcribed Text"):
        st.text(text)

    with st.expander("Error Messages"):
        st.write(f"Total Grammar Errors: {grammar_errors}")

    # **Cleanup Temporary Files**
    if os.path.exists(audio_path):
        os.remove(audio_path)
    if file_extension in ["mp4", "avi", "mov"] and os.path.exists(video_path):
        os.remove(video_path)
        for img in os.listdir("frames"):
            os.remove(f"frames/{img}")
