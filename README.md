# Twitter Sentiment Analysis (Streamlit)

Analyze recent posts from an X (Twitter) user for sentiment using VADER, visualize results, and export data. Built with Streamlit and Tweepy (v2 API).

## Features
- Input an X username to fetch recent tweets (excludes retweets/replies by default)
- VADER sentiment analysis with labels: Positive, Neutral, Negative
- Visualizations: overall distribution, time series, top words, and word cloud
- Keyword filter, date range filter, and CSV export
- Configure API credentials via `.env` or Streamlit `secrets.toml`

## Requirements
- Python 3.9+
- X (Twitter) API v2 credentials (at minimum a Bearer Token). For extended fields, use a Client ID/Secret for OAuth2 if needed.

## Quickstart

1. Clone the project and install dependencies:

```bash
pip install -r requirements.txt
```

2. Provide credentials either via environment variables (recommended during development):

Create a `.env` file (see `.env.example`):

```env
X_BEARER_TOKEN=YOUR_BEARER_TOKEN
```

Or use Streamlit secrets (recommended for deployment): create `.streamlit/secrets.toml`:

```toml
X_BEARER_TOKEN = "YOUR_BEARER_TOKEN"
```

3. Run the app:

```bash
streamlit run app.py
```

Open the provided local URL.

## Notes
- The app auto-downloads `vader_lexicon` on first run.
- If you see rate limits or missing tweets, adjust `max_results` or filters in the sidebar.
- This project uses only user-authorized endpoints that require a Bearer Token. For private/protected accounts you cannot fetch tweets.

## Project Structure
- `app.py` — Streamlit UI and orchestration
- `twitter_client.py` — Tweepy client and API calls
- `sentiment.py` — VADER sentiment analysis utilities
- `utils.py` — helpers for text cleaning and word cloud
- `.env.example` — example of environment configuration
- `.streamlit/secrets.toml` — secrets for Streamlit Cloud (not committed)

## License
MIT
