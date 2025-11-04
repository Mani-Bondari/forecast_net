from typing import Dict, List, Sequence, Tuple

import torch
from transformers import pipeline

# Priority: strong, robust 3-class English models with safetensors available.
MODEL_CANDIDATES = [
    ("cardiffnlp/twitter-roberta-base-sentiment-latest", "CardiffNLP RoBERTa-base (3-class, safetensors)"),
    ("j-hartmann/sentiment-roberta-large-english-3class", "RoBERTa large (3-class)"),
    ("cardiffnlp/twitter-xlm-roberta-base-sentiment", "XLM-RoBERTa base (3-class, multilingual)"),
]

CANONICAL = {"positive": "positive", "negative": "negative", "neutral": "neutral"}

_ANALYZER = None
_LABEL_LOOKUP = None


def load_sentiment_pipeline():
    """Load the first available advanced sentiment model from our candidates."""
    device = 0 if torch.cuda.is_available() else -1
    last_error = None

    for model_id, label in MODEL_CANDIDATES:
        print(f"Loading advanced sentiment model: {label}")
        try:
            analyzer = pipeline(
                task="sentiment-analysis",
                model=model_id,
                device=device,
                model_kwargs={},
                top_k=None,
                truncation=True,
                padding=True,
            )
            _ = analyzer("ok")  # Force model+tokenizer load; surfaces errors early.
            print(f"  -> using {model_id}")
            return analyzer
        except (OSError, ValueError) as exc:
            last_error = exc
            print(f"  -> unable to load {model_id} ({exc.__class__.__name__}): {exc}")

    raise RuntimeError(f"Could not load any sentiment model. Last error: {last_error}")


def _build_label_lookup(config) -> Dict[str, str]:
    raw = getattr(config, "id2label", {}) or {}
    lookup = {}

    for key, value in raw.items():
        value_l = str(value).lower()
        lookup[str(key).lower()] = value_l
        try:
            idx = int(key)
            lookup[str(idx)] = value_l
            lookup[f"label_{idx}"] = value_l
        except (ValueError, TypeError):
            pass
        lookup[value_l] = value_l

    return lookup


def _ensure_label_lookup(analyzer) -> Dict[str, str]:
    if analyzer is _ANALYZER and _LABEL_LOOKUP is not None:
        return _LABEL_LOOKUP
    return _build_label_lookup(analyzer.model.config)


def _canonicalize(label: str, lookup: Dict[str, str]) -> str:
    low = str(label).lower()
    mapped = lookup.get(low, low)

    if mapped in CANONICAL:
        return mapped
    if mapped.startswith("pos"):
        return "positive"
    if mapped.startswith("neg"):
        return "negative"
    if mapped.startswith("neu"):
        return "neutral"
    return mapped


def describe_strength(score: float) -> str:
    if score >= 0.75:
        return "very strong"
    if score >= 0.50:
        return "strong"
    if score >= 0.25:
        return "moderate"
    if score > 0.00:
        return "light"
    return "none"


def conclude_from_adjusted(adjusted: float, neutral: float, threshold: float = 0.10) -> str:
    if abs(adjusted) < threshold or neutral >= 0.60:
        return "Neutral"
    return "Positive" if adjusted > 0 else "Negative"


def get_sentiment_analyzer():
    global _ANALYZER, _LABEL_LOOKUP

    if _ANALYZER is None:
        _ANALYZER = load_sentiment_pipeline()
        _LABEL_LOOKUP = _build_label_lookup(_ANALYZER.model.config)

    return _ANALYZER


def _normalize_pipeline_output(raw_scores):
    if isinstance(raw_scores, list):
        if not raw_scores:
            return []
        first = raw_scores[0]
        if isinstance(first, dict):
            return raw_scores
        if isinstance(first, list):
            return first
    raise TypeError(f"Unexpected pipeline output format: {type(raw_scores)!r}")


def _aggregate_scores(score_entries, lookup: Dict[str, str]) -> Dict[str, float]:
    scores_by_label = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    for item in score_entries:
        label = _canonicalize(item["label"], lookup)
        if label in scores_by_label:
            scores_by_label[label] = float(item["score"])

    total = sum(scores_by_label.values())
    if total == 0.0:
        scores_by_label["neutral"] = 1.0
    elif not (0.999 <= total <= 1.001):
        for key in scores_by_label:
            scores_by_label[key] /= total

    return scores_by_label


def _build_result(clean_text: str, score_entries, lookup: Dict[str, str]):
    scores_by_label = _aggregate_scores(score_entries, lookup)

    p_pos = scores_by_label["positive"]
    p_neu = scores_by_label["neutral"]
    p_neg = scores_by_label["negative"]

    net_polarity = p_pos - p_neg
    adjusted_polarity = net_polarity * (1.0 - p_neu)
    primary_label = max(scores_by_label.items(), key=lambda kv: kv[1])[0].capitalize()
    conclusion = conclude_from_adjusted(adjusted_polarity, p_neu, threshold=0.10)

    return clean_text, primary_label, conclusion, scores_by_label, net_polarity, adjusted_polarity


def analyze_sentiment(text: str, analyzer=None) -> Tuple[str, str, str, Dict[str, float], float, float]:
    """
    Analyze sentiment for the supplied text and return a tuple containing:
      (clean_text, primary_label, conclusion, scores_by_label, net_polarity, adjusted_polarity)
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    clean_text = text.strip()
    analyzer = analyzer or get_sentiment_analyzer()
    lookup = _ensure_label_lookup(analyzer)

    raw_result = analyzer(clean_text or " ", truncation=True, padding=True)
    score_entries = _normalize_pipeline_output(raw_result)
    return _build_result(clean_text, score_entries, lookup)


def batch_analyze_sentiment(
    texts: Sequence[str],
    batch_size: int = 128,
    analyzer=None,
) -> List[Tuple[str, str, str, Dict[str, float], float, float]]:
    analyzer = analyzer or get_sentiment_analyzer()
    lookup = _ensure_label_lookup(analyzer)

    if not isinstance(texts, Sequence):
        raise TypeError("texts must be a sequence of strings")

    cleaned_texts: List[str] = []
    for text in texts:
        if not isinstance(text, str):
            raise TypeError("all items in texts must be strings")
        cleaned_texts.append(text.strip())

    raw_results = analyzer(cleaned_texts, batch_size=batch_size, truncation=True, padding=True)
    results: List[Tuple[str, str, str, Dict[str, float], float, float]] = []

    for clean_text, raw_scores in zip(cleaned_texts, raw_results):
        score_entries = _normalize_pipeline_output(raw_scores)
        results.append(_build_result(clean_text, score_entries, lookup))

    return results


def format_analysis(analysis: Tuple[str, str, str, Dict[str, float], float, float]) -> str:
    clean_text, primary_label, conclusion, scores_by_label, net_polarity, adjusted_polarity = analysis

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    lines = [
        "=" * 60,
        "sentiment analysis results",
        "=" * 60,
        f"\n Text: '{clean_text}'",
        f" Primary label: {primary_label}",
        (
            " Probabilities ->  "
            f"Positive: {pct(scores_by_label['positive'])} | "
            f"Neutral: {pct(scores_by_label['neutral'])} | "
            f"Negative: {pct(scores_by_label['negative'])}"
        ),
        f" Net polarity (pos - neg): {net_polarity:+.3f}",
        f" Adjusted polarity (net * (1 - neutral)): {adjusted_polarity:+.3f}",
        f" Conclusion: {conclusion}",
        "\n Breakdown by class:",
    ]

    for label, emoji in (("positive", "😍"), ("neutral", "😐"), ("negative", "😞")):
        score = scores_by_label[label]
        lines.append(f"   {emoji} {label.capitalize():8s}: {pct(score)}  ({describe_strength(score)})")

    return "\n".join(lines)

if __name__ == "__main__":
    user_text = input("insert text to give sentiment analysis on: ")
    analysis = analyze_sentiment(user_text)
    print("\n" + format_analysis(analysis))
