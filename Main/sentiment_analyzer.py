from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

def analyze_sentiment(text):
    """
    Analyzes sentiment of the given text.
    Returns sentiment label and compound score.
    """
    if not text:
        return "Neutral", 0.0

    try:
        scores = analyzer.polarity_scores(text)
        compound = scores['compound']

        if compound > 0.05:
            sentiment = "Positive"
        elif compound < -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return sentiment, compound
    except Exception as e:
        return f"Error in sentiment analysis: {e}", None
