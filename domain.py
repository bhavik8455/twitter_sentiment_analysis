from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import os
import json
import re
import pandas as pd


def _normalize_label(label: str) -> str:
	return re.sub(r"\s+", " ", label.strip()).title()


# Default built-in labels (fallback)
DEFAULT_DOMAIN_LABELS: List[str] = [
	"Politics",
	"Sports",
	"Education",
	"Technology",
	"Entertainment",
	"Business",
	"Health",
	"Science",
	"Travel",
	"Environment",
]


# Default built-in keyword lexicon (fallback)
KEYWORD_LEXICON: Dict[str, List[str]] = {
	"Politics": [
		"election", "policy", "government", "senate", "parliament", "minister", "president",
		"vote", "campaign", "politics", "congress", "diplomacy", "democracy",
	],
	"Sports": [
		"match", "game", "tournament", "league", "goal", "score", "coach", "athlete",
		"football", "soccer", "nba", "cricket", "tennis", "olympics",
	],
	"Education": [
		"school", "college", "university", "curriculum", "exam", "teacher", "student",
		"degree", "research", "scholarship", "classroom",
	],
	"Technology": [
		"ai", "artificial intelligence", "software", "hardware", "app", "coding", "programming",
		"cloud", "saas", "api", "data", "ml", "blockchain", "crypto", "startup",
	],
	"Entertainment": [
		"movie", "film", "music", "album", "song", "celebrity", "tv", "series", "hollywood",
		"bollywood", "concert", "festival",
	],
	"Business": [
		"market", "stock", "revenue", "profit", "merger", "acquisition", "startup", "funding",
		"economy", "inflation", "trade", "sales", "earnings",
	],
	"Health": [
		"hospital", "doctor", "nurse", "vaccine", "covid", "disease", "medicine", "mental health",
		"fitness", "diet", "wellness",
	],
	"Science": [
		"research", "study", "experiment", "theory", "physics", "chemistry", "biology",
		"astronomy", "space", "quantum", "lab",
	],
	"Travel": [
		"flight", "airport", "hotel", "tour", "trip", "journey", "visa", "tourism", "itinerary",
	],
	"Environment": [
		"climate", "emissions", "sustainability", "wildlife", "conservation", "pollution",
		"renewable", "recycling", "ecosystem",
	],
}


def _load_external_lexicon(path: Optional[str] = None) -> Optional[Dict[str, List[str]]]:
	"""Load a domain->keywords mapping from JSON if available.

	Expected JSON format: { "Domain Name": ["keyword1", "keyword2", ...], ... }
	"""
	try:
		candidate_path = path or os.path.join(os.getcwd(), "domain_classification.json")
		if not os.path.exists(candidate_path):
			return None
		with open(candidate_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		if not isinstance(data, dict):
			return None
		lex: Dict[str, List[str]] = {}
		for raw_label, keywords in data.items():
			if not isinstance(raw_label, str):
				continue
			label = _normalize_label(raw_label)
			if isinstance(keywords, list):
				# Lowercase keywords for matching
				kw = [str(k).strip().lower() for k in keywords if str(k).strip()]
				if kw:
					lex[label] = kw
		return lex or None
	except Exception:
		# Fail silently and fall back to built-in lexicon
		return None


# If the user provided a JSON lexicon, override defaults at import time
_EXTERNAL = _load_external_lexicon()
if _EXTERNAL:
	KEYWORD_LEXICON = _EXTERNAL
	DEFAULT_DOMAIN_LABELS = list(KEYWORD_LEXICON.keys())


def _keyword_score(text: str, keywords: Iterable[str]) -> int:
	# Simple substring count on lowercased text, supports multi-word phrases
	text_lc = text.lower()
	return sum(1 for kw in keywords if kw in text_lc)


def classify_domains_keyword(
	df: pd.DataFrame,
	candidate_labels: Optional[List[str]] = None,
)	-> pd.DataFrame:
	labels = [
		_normalize_label(lbl)
		for lbl in (candidate_labels if candidate_labels else DEFAULT_DOMAIN_LABELS)
	]
	lexicon = {lbl: KEYWORD_LEXICON.get(lbl, []) for lbl in labels}

	def pick_label(text: str) -> Tuple[str, float]:
		scores = {lbl: _keyword_score(text, lexicon[lbl]) for lbl in labels}
		best_label, best_score = max(scores.items(), key=lambda kv: kv[1])
		if best_score == 0:
			return ("Other", 0.0)
		return (best_label, float(best_score))

	res = df.copy()
	assignments = res["text_clean"].astype(str).apply(pick_label)
	res["domain_label"] = assignments.apply(lambda x: x[0])
	res["domain_confidence"] = assignments.apply(lambda x: x[1])
	return res


def _get_zero_shot_pipeline():
	try:
		from transformers import pipeline  # type: ignore
		return pipeline(
			"zero-shot-classification",
			model="facebook/bart-large-mnli",
			device=-1,
		)
	except Exception as e:  # pragma: no cover
		raise RuntimeError(
			"Zero-shot classifier requires 'transformers' (and a backend like PyTorch)."
		) from e


def classify_domains_zero_shot(
	df: pd.DataFrame,
	candidate_labels: Optional[List[str]] = None,
)	-> pd.DataFrame:
	labels = [
		_normalize_label(lbl)
		for lbl in (candidate_labels if candidate_labels else DEFAULT_DOMAIN_LABELS)
	]
	zs = _get_zero_shot_pipeline()

	def infer(text: str) -> Tuple[str, float]:
		if not text or not text.strip():
			return ("Other", 0.0)
		res = zs(text, labels, multi_label=False)
		# The pipeline returns labels sorted by score
		return (res["labels"][0], float(res["scores"][0]))

	res_df = df.copy()
	assignments = res_df["text_clean"].astype(str).apply(infer)
	res_df["domain_label"] = assignments.apply(lambda x: x[0])
	res_df["domain_confidence"] = assignments.apply(lambda x: x[1])
	return res_df


def analyze_domains(
	df: pd.DataFrame,
	candidate_labels: Optional[List[str]] = None,
	method: str = "keyword",
) -> pd.DataFrame:
	method_norm = (method or "keyword").strip().lower()
	if method_norm == "zero-shot":
		return classify_domains_zero_shot(df, candidate_labels)
	return classify_domains_keyword(df, candidate_labels)


