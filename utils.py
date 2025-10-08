import re
from wordcloud import WordCloud
from PIL import Image


URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
	text = text or ""
	text = URL_RE.sub(" ", text)
	text = MENTION_RE.sub(" ", text)
	text = HASHTAG_RE.sub("", text)
	text = re.sub(r"&amp;", "&", text)
	text = re.sub(r"&lt;", "<", text)
	text = re.sub(r"&gt;", ">", text)
	text = MULTISPACE_RE.sub(" ", text)
	return text.strip().lower()


def generate_wordcloud_image(text: str) -> Image.Image:
	if not text.strip():
		text = "neutral positive negative"
	wc = WordCloud(width=1200, height=600, background_color="white").generate(text)
	return wc.to_image()
