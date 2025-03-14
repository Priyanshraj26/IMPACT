import streamlit as st
import nltk
import whisper
from nltk.tokenize import sent_tokenize
import language_tool_python
import numpy as np
import io
import soundfile as sf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('punkt')

# Load Whisper model
model = whisper.load_model("medium")  # You can use "tiny", "base", "medium", or "large"

# Initialize Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Streamlit app
st.title("Speech to Text and Grammar Checker")

# Upload audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file is not None:
    # Initialize the progress bar
    progress_bar = st.progress(0)

    # Read the uploaded file into a numpy array
    audio_bytes = uploaded_file.read()
    audio, samplerate = sf.read(io.BytesIO(audio_bytes))

    # Display audio player
    st.audio(audio_bytes, format='audio/wav')

    # Update progress bar
    progress_bar.progress(20)

    # Ensure the audio data is in the correct format (float32)
    audio = audio.astype(np.float32)

    # Update progress bar
    progress_bar.progress(40)

    # Transcribe audio file
    if 'transcribed_text' not in st.session_state:
        result = model.transcribe(audio)
        st.session_state.transcribed_text = result["text"]

    text = st.session_state.transcribed_text

    # Update progress bar
    progress_bar.progress(60)

    # Calculate duration of the audio in seconds
    duration_seconds = len(audio) / samplerate

    # Calculate WPM
    def calculate_wpm(transcript, duration_seconds):
        words = transcript.split()  # Tokenize words
        total_words = len(words)
        duration_minutes = duration_seconds / 60  # Convert seconds to minutes
        wpm = total_words / duration_minutes if duration_minutes > 0 else 0
        return round(wpm, 2)

    wpm = calculate_wpm(text, duration_seconds)

    # Update progress bar
    progress_bar.progress(80)

    # Tokenize sentences
    sentences = sent_tokenize(text)

    # Initialize language tool
    tool = language_tool_python.LanguageTool('en-US')

    # Define rules to ignore
    IGNORE_RULES = {
        "MORFOLOGIK_RULE_EN_US",  # Spelling mistakes
        "COMMA_COMPOUND_SENTENCE",  # Missing comma before 'and'
        "THANKS_SENT_END_COMMA",  # Comma after "Thank you"
        "ALL_OF_THE",  # Minor rewording suggestions
        "AND_SO_ONE"  # "and so on" suggestion
    }

    # Calculate grammar errors
    if 'error_messages' not in st.session_state:
        grammar_errors = 0
        error_messages = []
        for sentence in sentences:
            matches = tool.check(sentence)
            grammar_errors += len(matches)  # Count total grammar mistakes
            for match in matches:
                error_messages.append(f"Error in: '{sentence}' → {match.ruleId}: {match.message}")
        st.session_state.error_messages = error_messages
        st.session_state.grammar_errors = grammar_errors

    error_messages = st.session_state.error_messages
    grammar_errors = st.session_state.grammar_errors

    # Function to calculate grammar score per sentence
    def calculate_sentence_score(sentence):
        matches = tool.check(sentence)
        major_errors = 0
        minor_errors = 0
        for match in matches:
            if match.ruleId not in IGNORE_RULES:
                if "agreement" in match.ruleId.lower() or "tense" in match.ruleId.lower():
                    major_errors += 1  # Higher weight for critical grammar issues
                else:
                    minor_errors += 1  # Lower weight for minor grammar mistakes
        total_words = len(sentence.split())  # Word count in sentence
        if total_words == 0:
            return 10.0  # No words → perfect score
        # Weighted formula: Major errors count 2x more
        weighted_errors = (major_errors * 2) + minor_errors
        score = max(0, 10 - ((weighted_errors / total_words) * 100))
        return round(score, 2)

    # Function to process multiple sentences
    def calculate_grammar_score(sentences):
        sentence_scores = [calculate_sentence_score(sentence) for sentence in sentences]
        if len(sentence_scores) == 0:
            return 10.0  # No sentences → perfect score
        # Final grammar score (average of all sentence scores)
        final_score = sum(sentence_scores) / len(sentence_scores)
        return round(final_score, 2)

    # Compute final grammar score
    final_grammar_score = calculate_grammar_score(sentences)

    # Analyze sentiment
    def analyze_sentiment(transcript):
        sentiment_score = analyzer.polarity_scores(transcript)
        compound_score = sentiment_score['compound']
        sentiment = "Positive" if compound_score > 0.05 else "Negative" if compound_score < -0.05 else "Neutral"
        return sentiment, compound_score

    sentiment, compound_score = analyze_sentiment(text)

    # Display results in three columns
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<h2 style='font-size: 24px;'>Words Per Minute (WPM)</h2>", unsafe_allow_html=True)
        st.write(f"{wpm} WPM")
    with col2:
        st.markdown("<h2 style='font-size: 24px;'>Sentiment Analysis</h2>", unsafe_allow_html=True)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Compound Score: {compound_score}")
    with col3:
        st.markdown("<h2 style='font-size: 24px;'>Final Grammar Score</h2>", unsafe_allow_html=True)
        st.write(f"{final_grammar_score}/10")

    # Display transcribed text in an expander
    with st.expander("Transcribed Text"):
        st.markdown("<h3 style='font-size: 18px;'>Transcribed Text</h3>", unsafe_allow_html=True)
        st.text(text)

    # Display error messages
    with st.expander("Error Messages"):
        st.markdown("<h3 style='font-size: 18px;'>Error Messages</h3>", unsafe_allow_html=True)
        st.write(f"Total Grammar Errors: {grammar_errors}")
        for error in error_messages:
            st.text(error)

    # Update progress bar to 100%
    progress_bar.progress(100)

    #UPDATE HERE
    #updating on main rn 