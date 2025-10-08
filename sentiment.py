from typing import Tuple

import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Ensure VADER lexicon is available
try:
	nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
	nltk.download('vader_lexicon')


def score_to_label(compound: float) -> str:
	if compound >= 0.05:
		return "Positive"
	elif compound <= -0.05:
		return "Negative"
	return "Neutral"


def analyze_sentiments(df: pd.DataFrame) -> pd.DataFrame:
	"""Add sentiment columns to a DataFrame with a 'text_clean' column."""
	sia = SentimentIntensityAnalyzer()
	scores = df["text_clean"].astype(str).apply(sia.polarity_scores)
	compound = scores.apply(lambda s: s["compound"])  # type: ignore[index]
	df = df.copy()
	df["compound"] = compound
	df["sentiment_label"] = df["compound"].apply(score_to_label)
	return df
