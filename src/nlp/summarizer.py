"""Extract full article text and summarize to fit FinBERT's 512 token limit."""

from newspaper import Article
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
summarizer = TextRankSummarizer()
MAX_TOKENS = 480  # leave some margin under 512


def fetch_article_text(url: str) -> str:
    """Download and extract full article text from a URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception:
        return ""


def summarize_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    """Extract the most important sentences that fit within the token limit."""
    if not text or not text.strip():
        return ""

    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    if token_count <= max_tokens:
        return text

    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    ranked_sentences = summarizer(parser.document, sentences_count=len(parser.document.sentences))

    result_sentences = []
    current_tokens = 0
    for sentence in ranked_sentences:
        s = str(sentence)
        s_tokens = len(tokenizer.encode(s, add_special_tokens=False))
        if current_tokens + s_tokens > max_tokens:
            break
        result_sentences.append(s)
        current_tokens += s_tokens

    return " ".join(result_sentences) if result_sentences else text[:1500]


def get_article_summary(url: str, title: str = "", description: str = "") -> str:
    """Full pipeline: fetch article → summarize → return text ready for FinBERT."""
    full_text = fetch_article_text(url)

    if not full_text:
        # Fallback to title + description if scraping fails
        return f"{title}. {description}".strip(". ")

    return summarize_to_tokens(full_text)


if __name__ == "__main__":
    # Quick test
    test_url = "https://www.cnbc.com"
    text = fetch_article_text(test_url)
    print(f"Fetched {len(text)} chars")
    if text:
        summary = summarize_to_tokens(text)
        tokens = len(tokenizer.encode(summary, add_special_tokens=False))
        print(f"Summary: {tokens} tokens")
        print(summary[:300])
