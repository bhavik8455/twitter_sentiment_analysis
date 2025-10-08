import os
import json
import time
from datetime import datetime
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from twitter_client import TwitterClient
from twitter_client import RateLimitError
from sentiment import analyze_sentiments
from utils import generate_wordcloud_image, clean_text

# Load .env if present
load_dotenv()

st.set_page_config(page_title="X (Twitter) Sentiment Analysis", layout="wide")

# Sidebar - Configuration
with st.sidebar:
	st.header("Settings")
	st.write("Provide an X username to analyze their recent posts.")

	username = st.text_input("X Username (without @)", value="")
	max_results = st.slider("Max tweets to fetch", min_value=5, max_value=15, step=1, value=10)
	exclude_replies = st.checkbox("Exclude replies", value=True)
	exclude_retweets = st.checkbox("Exclude retweets", value=True)
	start_date = st.date_input("Start date (optional)", value=None)
	end_date = st.date_input("End date (optional)", value=None)

	keyword_filter = st.text_input("Keyword contains (optional)", value="")
	analyze_button = st.button("Analyze")

# Main Title
st.title("X (Twitter) User Sentiment Analysis")

# Instantiate API Client (uses env/secrets)
client: Optional[TwitterClient] = None
try:
	client = TwitterClient()
except Exception as e:
	st.error(f"Failed to initialize X API client: {e}")

@st.cache_data(show_spinner=False, ttl=900)
def fetch_and_prepare(username: str, max_results: int, exclude_replies: bool, exclude_retweets: bool, start_date: Optional[datetime], end_date: Optional[datetime]) -> pd.DataFrame:
	if client is None:
		raise RuntimeError("API client not available")

	user = client.get_user_by_username(username)
	if user is None:
		return pd.DataFrame()

	user_id = user.id
	# Fetch tweets
	tweets = client.get_user_tweets(
		user_id=user_id,
		max_results=max_results,
		exclude_replies=exclude_replies,
		exclude_retweets=exclude_retweets,
		start_time=start_date,
		end_time=end_date,
	)

	if not tweets:
		return pd.DataFrame()

	records = []
	for t in tweets:
		records.append({
			"id": t.id,
			"created_at": t.created_at,
			"text": t.text,
			"like_count": getattr(t, "public_metrics", {}).get("like_count", None) if hasattr(t, "public_metrics") else None,
			"retweet_count": getattr(t, "public_metrics", {}).get("retweet_count", None) if hasattr(t, "public_metrics") else None,
			"reply_count": getattr(t, "public_metrics", {}).get("reply_count", None) if hasattr(t, "public_metrics") else None,
			"quote_count": getattr(t, "public_metrics", {}).get("quote_count", None) if hasattr(t, "public_metrics") else None,
		})

	df = pd.DataFrame.from_records(records)
	if df.empty:
		return df

	# Clean text and basic filtering
	df["text_clean"] = df["text"].astype(str).apply(clean_text)
	return df


def render_overview(df: pd.DataFrame):
	st.subheader("Overview")
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		st.metric("Tweets", len(df))
	with col2:
		st.metric("Avg Likes", round(df["like_count"].dropna().mean(), 2) if "like_count" in df else 0)
	with col3:
		st.metric("Avg Retweets", round(df["retweet_count"].dropna().mean(), 2) if "retweet_count" in df else 0)
	with col4:
		st.metric("Avg Replies", round(df["reply_count"].dropna().mean(), 2) if "reply_count" in df else 0)


def render_charts(df: pd.DataFrame):
	st.subheader("Sentiment Distribution")
	counts = df["sentiment_label"].value_counts().reset_index()
	counts.columns = ["sentiment", "count"]
	fig_bar = px.bar(counts, x="sentiment", y="count", color="sentiment", color_discrete_map={
		"Positive": "#2ecc71",
		"Neutral": "#95a5a6",
		"Negative": "#e74c3c",
	})
	st.plotly_chart(fig_bar, use_container_width=True)

	st.subheader("Sentiment Over Time")
	df_time = df.copy()
	df_time["date"] = pd.to_datetime(df_time["created_at"]).dt.date
	series = df_time.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
	fig_area = px.area(series, x="date", y="count", color="sentiment_label", color_discrete_map={
		"Positive": "#2ecc71",
		"Neutral": "#95a5a6",
		"Negative": "#e74c3c",
	})
	st.plotly_chart(fig_area, use_container_width=True)

	st.subheader("Word Cloud")
	image = generate_wordcloud_image(" ".join(df.loc[df["sentiment_label"] != "Neutral", "text_clean"].tolist()) or " ")
	st.image(image, caption="Word cloud of non-neutral tweets", use_column_width=True)


def render_table(df: pd.DataFrame):
	st.subheader("Tweets")
	st.dataframe(df[["created_at", "sentiment_label", "compound", "text"]].sort_values("created_at", ascending=False), use_container_width=True, height=400)

	# Export full records (all columns) as JSON
	records = df.sort_values("created_at", ascending=False).to_dict(orient="records")
	json_str = json.dumps(records, ensure_ascii=False, indent=2, default=str)
	st.download_button("Download JSON", data=json_str.encode("utf-8"), file_name="tweets_sentiment.json", mime="application/json")


if analyze_button:
	if not username.strip():
		st.warning("Please enter a username.")
		st.stop()

	with st.spinner("Fetching and analyzing tweets..."):
		try:
			df = fetch_and_prepare(
				username=username.strip().lstrip("@"),
				max_results=max_results,
				exclude_replies=exclude_replies,
				exclude_retweets=exclude_retweets,
				start_date=start_date,
				end_date=end_date,
			)
		except RateLimitError as rle:
			msg = "Rate limit reached. Please try again later."
			if rle.retry_after_seconds is not None:
				msg += f" Suggested wait: ~{rle.retry_after_seconds} seconds."
			st.warning(msg)
			st.stop()
		except Exception as e:
			st.error("Could not fetch tweets right now. Please try again in a few minutes.")
			st.stop()

	if df.empty:
		st.info("No tweets found for this user with current filters.")
		st.stop()

	# Keyword filtering
	if keyword_filter.strip():
		mask = df["text_clean"].str.contains(keyword_filter.strip().lower())
		df = df[mask]
		if df.empty:
			st.info("No tweets match the keyword filter.")
			st.stop()

	# Sentiment analysis
	df = analyze_sentiments(df)

	render_overview(df)
	render_charts(df)
	render_table(df)

else:
	st.info("Enter a username and click Analyze to begin.")
