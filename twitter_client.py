from datetime import datetime
from typing import List, Optional
import os

import tweepy
import streamlit as st
from tweepy.errors import TooManyRequests, TweepyException


class RateLimitError(Exception):
	"""Raised when the X API rate limit is hit. Optionally includes seconds to wait."""

	def __init__(self, message: str = "Rate limit exceeded", retry_after_seconds: Optional[int] = None) -> None:
		super().__init__(message)
		self.retry_after_seconds = retry_after_seconds


class TwitterClient:
	"""Simple wrapper around Tweepy Client for X (Twitter) API v2."""

	def __init__(self) -> None:
		# Prefer environment variable over Streamlit secrets to avoid placeholder overrides
		bearer_token = os.getenv("X_BEARER_TOKEN") or self._get_secret("X_BEARER_TOKEN")
		# Guard against placeholder values
		if bearer_token and bearer_token.strip().upper() in {"YOUR_BEARER_TOKEN_HERE", "YOUR_BEARER_TOKEN"}:
			bearer_token = None
		if not bearer_token:
			raise ValueError("X_BEARER_TOKEN is required. Set it in .env or Streamlit secrets.")

		self.client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)

	def _get_secret(self, key: str) -> Optional[str]:
		try:
			return st.secrets.get(key)  # type: ignore[attr-defined]
		except Exception:
			return None

	def get_user_by_username(self, username: str) -> Optional[tweepy.User]:
		try:
			resp = self.client.get_user(username=username)
			return resp.data
		except TooManyRequests as e:
			retry_after = _retry_after_from_response(e)
			raise RateLimitError(retry_after_seconds=retry_after)
		except TweepyException as e:
			raise RuntimeError(f"Error fetching user '{username}'") from e

	def get_user_tweets(
		self,
		user_id: str,
		max_results: int = 200,
		exclude_replies: bool = True,
		exclude_retweets: bool = True,
		start_time: Optional[datetime] = None,
		end_time: Optional[datetime] = None,
	) -> List[tweepy.Tweet]:
		"""Fetch recent tweets for a user, respecting basic filters.

		Note: The API returns up to 100 per call; we paginate until we reach max_results or run out.
		"""
		collected: List[tweepy.Tweet] = []
		exclude = []
		if exclude_replies:
			exclude.append("replies")
		if exclude_retweets:
			exclude.append("retweets")

		pagination_token: Optional[str] = None
		remaining = max(1, max_results)

		while remaining > 0:
			limit = min(100, remaining)
			resp = self.client.get_users_tweets(
				id=user_id,
				max_results=limit,
				exclude=exclude or None,
				start_time=start_time,
				end_time=end_time,
				tweet_fields=["created_at", "public_metrics"],
				pagination_token=pagination_token,
			)

			if resp.data is None:
				break

			collected.extend(resp.data)
			remaining -= len(resp.data)
			pagination_token = resp.meta.get("next_token") if resp.meta else None
			if not pagination_token:
				break

		return collected


def _retry_after_from_response(e: Exception) -> Optional[int]:
	"""Extract seconds-until-reset from Tweepy exception response headers if available."""
	try:
		resp = getattr(e, "response", None)
		if not resp or not hasattr(resp, "headers"):
			return None
		headers = resp.headers or {}
		# Prefer direct Retry-After if present
		retry_after = headers.get("retry-after") or headers.get("Retry-After")
		if retry_after and str(retry_after).isdigit():
			return int(retry_after)
		# Else try x-rate-limit-reset epoch delta
		reset_epoch = headers.get("x-rate-limit-reset") or headers.get("X-Rate-Limit-Reset")
		if reset_epoch and str(reset_epoch).isdigit():
			from time import time as now
			return max(0, int(reset_epoch) - int(now()))
		return None
	except Exception:
		return None
